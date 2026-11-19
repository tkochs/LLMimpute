import importlib.util
import sys
import uuid
from pathlib import Path


class SchemaManager:

    @staticmethod
    def create(generator, experiment_id):

        schema_path = (
            Path("generated_schemas")
            / experiment_id
            / "schema.py"
        )

        generator.save(str(schema_path))

        module_name = f"schema_{uuid.uuid4().hex}"

        spec = importlib.util.spec_from_file_location(
            module_name,
            schema_path
        )

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return module.ProductRow