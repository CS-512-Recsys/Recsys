class Trainer(object):
    def __init__(self, model, device,loss_fn=None, optimizer=None, scheduler=None,artifacts_loc=None,exp_tracker=None):

        # Set params
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.store_loc = artifacts_loc
        self.exp_tracker = exp_tracker

    def train_step(self, dataloader):
        """Train step."""
        # Set model to train mode
        self.model.train()
        loss = 0.0

        # Iterate over train batches
        for i, batch in enumerate(dataloader):
            #batch = [item.to(self.device) for item in batch]  # Set device
            inputs,targets = batch
            inputs = [item.to(self.device) for item in inputs]
            targets = targets.to(self.device)
            #inputs, targets = batch[:-1], batch[-1]
            #import pdb;pdb.set_trace()
            self.optimizer.zero_grad()  # Reset gradients
            z = self.model(inputs)  # Forward pass
            targets = targets.reshape(z.shape)
            J = self.loss_fn(z.float(), targets.float())  # Define loss
            J.backward()  # Backward pass
            self.optimizer.step()  # Update weights

            # Cumulative Metrics
            loss += (J.detach().item() - loss) / (i + 1)

        return loss

    def eval_step(self, dataloader):
        """Validation or test step."""
        # Set model to eval mode
        self.model.eval()
        loss = 0.0
        y_trues, y_probs = [], []

        # Iterate over val batches
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                inputs,y_true = batch
                inputs = [item.to(self.device) for item in inputs]
                y_true = y_true.to(self.device).float()

                # Step
                z = self.model(inputs).float()  # Forward pass
                y_true = y_true.reshape(z.shape)
                J = self.loss_fn(z, y_true).item()

                # Cumulative Metrics
                loss += (J - loss) / (i + 1)

                # Store outputs
                y_prob = z.cpu().numpy()
                y_probs.extend(y_prob)
                y_trues.extend(y_true.cpu().numpy())

        return loss, np.vstack(y_trues), np.vstack(y_probs)

    def predict_step(self, dataloader):
        """Prediction step."""
        # Set model to eval mode
        self.model.eval()
        y_probs = []

        # Iterate over val batches
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):

                # Forward pass w/ inputs
                inputs, targets = batch
                z = self.model(inputs).float()

                # Store outputs
                y_prob = z.cpu().numpy()
                y_probs.extend(y_prob)

        return np.vstack(y_probs)
    def train(self, num_epochs, patience, train_dataloader, val_dataloader,
              tolerance=1e-5):
        best_val_loss = np.inf
        training_stats = defaultdict(list)
        for epoch in tqdm(range(num_epochs)):
            # Steps
            train_loss = self.train_step(dataloader=train_dataloader)
            val_loss, _, _ = self.eval_step(dataloader=val_dataloader)
            #store stats
            training_stats['epoch'].append(epoch)
            training_stats['train_loss'].append(train_loss)
            training_stats['val_loss'].append(val_loss)
            #log-stats
            if self.exp_tracker == 'wandb':
                log_metrics = {'epoch':epoch,'train_loss':train_loss,'val_loss':val_loss}
                wandb.log(log_metrics,step=epoch)
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss - tolerance:
                best_val_loss = val_loss
                best_model = self.model
                _patience = patience  # reset _patience
            else:
                _patience -= 1
            if not _patience:  # 0
                print("Stopping early!")
                break

            # Tracking
            #mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            # Logging
            if epoch%10 == 0:
                print(
                    f"Epoch: {epoch+1} | "
                    f"train_loss: {train_loss:.5f}, "
                    f"val_loss: {val_loss:.5f}, "
                    f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, "
                    f"_patience: {_patience}"
                )
        if self.store_loc:
            pd.DataFrame(training_stats).to_csv(self.store_loc/'training_stats.csv',index=False)
        return best_model, best_val_loss