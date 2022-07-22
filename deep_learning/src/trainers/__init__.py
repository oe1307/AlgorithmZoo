from .metric_trainer import MetricLearningTrainer


def get_trainer(model, config, dataloader):
    trainer = MetricLearningTrainer(model, config, dataloader)
    return trainer
