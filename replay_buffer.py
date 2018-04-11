import random
from collections import namedtuple, deque

class ReplayBuffer():
    def __init__(self, max_size, batch_size):
        """
        Creates an instance of the Replay buffer which is used for storing tuple experiences.

        Args:
            max_size (int): Buffer maximum size.
            batch_size (int): Batch size.
        """
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def add(self, state, action, reward, next_state, done):
        """
        Adds an experience to the buffer.
        """
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self):
        """
        Samples a random batch from the buffer.
        """
        return random.sample(self.buffer, k=self.batch_size)
        
    def __len__(self):
        """Return the current size of internal buffer."""
        return len(self.buffer)