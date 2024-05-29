class Logger:
    def __init__(self, push_interval=1):
        self.data = {}
        self.push_interval = push_interval

    def log(self, key, value, epoch):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append((epoch, value))

    def end_of_epoch(self, epoch):
        if epoch % self.push_interval == 0:
            self.push()

    def push(self):
        raise NotImplementedError("Push method should be implemented by subclasses")

class TerminalLogger(Logger):
    def push(self):
        for key, values in self.data.items():
            for epoch, value in values:
                print(f"Epoch {epoch}: {key} = {value}")
        self.data = {}

class WandbLogger(Logger):
    def __init__(self, project, entity, config, push_interval=1):
        super().__init__(push_interval)
        import wandb
        self.wandb = wandb
        self.wandb.init(project=project, entity=entity, config=config)

    def push(self):
        for key, values in self.data.items():
            for epoch, value in values:
                self.wandb.log({key: value, 'epoch': epoch})
        self.data = {}
