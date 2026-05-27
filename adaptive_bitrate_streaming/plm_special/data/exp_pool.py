
class ExperiencePool:
    """
    Experience pool for collecting trajectories.
    Optional ``teacher_logits`` per step: shape (BITRATE_LEVELS,) logits or probabilities.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.teacher_logits = []

    def add(self, state, action, reward, done, teacher_logits=None):
        self.states.append(state)  # sometime state is also called obs (observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        if teacher_logits is not None:
            self.teacher_logits.append(teacher_logits)

    @property
    def has_teacher_logits(self):
        return len(self.teacher_logits) > 0 and len(self.teacher_logits) == len(self.states)

    def __len__(self):
        return len(self.states)

