import tensorflow as tf
import numpy as np
import scipy.misc
import threading
from io import BytesIO


class Logger:
    def __init__(self,
                 log_interval=50,
                 validation_interval=200,
                 generate_interval=500,
                 trainer=None,
                 generate_function=None):
        self.trainer = trainer
        self.log_interval = log_interval
        self.validation_interval = validation_interval
        self.generate_interval = generate_interval
        self.accumulated_loss = 0
        self.generate_function = generate_function
        self.current_step = 0
        if self.generate_function is not None:
            self.generate_thread = threading.Thread(target=self.generate_function)
            self.generate_function.daemon = True

    def log(self, current_step, current_loss):
        self.current_step = current_step 
        self.accumulated_loss += current_loss
        if current_step % self.log_interval == 0:
            self.log_loss(current_step)
            self.accumulated_loss = 0
        if current_step % self.validation_interval == 0:
            self.validate(current_step)
        if current_step % self.generate_interval == 0:
            self.generate(current_step)

    def log_loss(self, current_step):
        avg_loss = self.accumulated_loss / self.log_interval
        print("loss at step " + str(current_step) + ": " + str(avg_loss))

    def validate(self, current_step):
        avg_loss, avg_accuracy = self.trainer.validate()
        print("validation loss: " + str(avg_loss))
        print("validation accuracy: " + str(avg_accuracy * 100) + "%")

class TensorboardLogger(Logger):
    def __init__(self,
                 log_interval=50,
                 validation_interval=200,
                 generate_interval=500,
                 trainer=None,
                 generate_function=None,
                 log_dir='logs'):
        super().__init__(log_interval, validation_interval, generate_interval, trainer, generate_function)
        self.writer = tf.summary.create_file_writer(log_dir)  # TF 2.x

    def log_loss(self, current_step):
        avg_loss = self.accumulated_loss / self.log_interval
        self.scalar_summary('loss', avg_loss, current_step)
        print(f"\n[Train] Step {current_step:05d} | Loss: {avg_loss:.4f}")

        for tag, value in self.trainer.model.named_parameters():
            tag = tag.replace('.', '/')
            self.histo_summary(tag, value.data.cpu().numpy(), current_step)
            if value.grad is not None:
                self.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), current_step)

    def validate(self, current_step):
        avg_loss, avg_accuracy = self.trainer.validate()
        self.scalar_summary('validation loss', avg_loss, current_step) 
        self.scalar_summary('validation accuracy', avg_accuracy, current_step)
        print(f"[Validation] Step {current_step:05d} | Loss: {avg_loss:.4f} | Accuracy: {avg_accuracy*100:.2f}%")

    def log_audio(self, step):
        samples = self.generate_function()
        tf_samples = tf.convert_to_tensor(samples)
        self.audio_summary('audio sample', tf_samples, step, sr=16000)

    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def audio_summary(self, tag, sample, step, sr=16000):
        with self.writer.as_default():
            tf.summary.audio(tag, sample, sample_rate=sr, step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=200):
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step)
            self.writer.flush()
