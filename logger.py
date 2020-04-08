from pytorch_lightning.logging import LightningLoggerBase, rank_zero_only

class Logger(LightningLoggerBase):

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    def log_metrics(self, metrics, step_num):
        self.logger.experiment.
        
        
    def save(self):
        pass

    def finalize(self, status):
        pass