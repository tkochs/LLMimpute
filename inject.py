from swiss_cheese import MCAR, MAR, MNAR
import yaml
import pandas as pd

class MissingValueGenerator:

    def __init__(self, config):

        missing_cfg = config["missingness"]

        self.mechanism = missing_cfg["mechanism"]
        self.alpha = missing_cfg["missing_rate"]

        self.mean = missing_cfg["mean"]
        self.variance = missing_cfg["variance"]
        self.mode = missing_cfg["mode"]
        self.seed = missing_cfg["seed"]

    def inject(self, df):

        # Create a copy so that the original dataset
        # remains unchanged for evaluation.
        df = df.copy()

        if self.mechanism == "MCAR":

            df_missing = MCAR(
                random_seed=self.seed
            )(df, self.alpha)

        elif self.mechanism == "MAR":

            df_missing = MAR(
                mean=self.mean,
                variance=self.variance,
                mode=self.mode,
                random_seed=self.seed
            )(df, self.alpha)

        elif self.mechanism == "MNAR":

            df_missing = MNAR(
                mean=self.mean,
                variance=self.variance,
                mode=self.mode,
                random_seed=self.seed
            )(df, self.alpha)

        else:

            raise ValueError(
                f"Unknown mechanism: {self.mechanism}"
            )

        return df_missing