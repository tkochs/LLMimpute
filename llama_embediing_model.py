import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings


class EmbeddingModel(Embeddings):

    def __init__(self,model_name="meta-llama/Llama-3.1-8B-Instruct",device_map="auto",torch_dtype=torch.bfloat16,max_length=512,batch_size=16,):
        self.max_length = max_length
        self.batch_size = batch_size

        print("Loading Llama tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Llama has no dedicated padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading Llama embedding model...")
        self.model = AutoModel.from_pretrained(model_name,torch_dtype=torch_dtype,device_map=device_map,)

        self.model.eval()
        print("Loaded Llama embedding model")


    @torch.no_grad()
    def _embed_batch(self, texts):

        inputs = self.tokenizer(texts,padding=True,truncation=True,max_length=self.max_length,return_tensors="pt",)

        inputs = {k: v.to(self.model.device)for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # [batch_size, sequence_length, 4096]
        hidden_states = outputs.last_hidden_state

        # [batch_size, sequence_length]
        attention_mask = inputs["attention_mask"]

        # [batch_size, sequence_length, 1]
        attention_mask = attention_mask.unsqueeze(-1)

        # Make PAD token hidden states zero
        masked_hidden = hidden_states * attention_mask

        # Add all real token vectors
        summed = masked_hidden.sum(dim=1)

        # Count number of real tokens
        counts = attention_mask.sum(dim=1).clamp(min=1)

        # Mean pooling
        embeddings = summed / counts

        # L2 normalization
        embeddings = F.normalize(embeddings,p=2,dim=1)
        return embeddings.float().cpu()


    def embed_documents(self, texts):
        """
        Used by Chroma when storing documents.
        """

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):

            batch = texts[i:i + self.batch_size]

            embeddings = self._embed_batch(batch)

            all_embeddings.extend(
                embeddings.tolist()
            )

        return all_embeddings


    def embed_query(self, text):
        """
        Used by Chroma when embedding retrieval query.
        """

        embedding = self._embed_batch([text])

        return embedding[0].tolist()