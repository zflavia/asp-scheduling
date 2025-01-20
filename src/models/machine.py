class Machine:
    def __init__(self):
        self.intervals = []

    def add_interval(self, index, task):
        if index == -1:
            self.intervals.append(task)
        else:
            self.intervals.insert(index, task)

    def add_last_interval(self, task):
        self.intervals.append(task)

    def get_duration(self, task):
        return self.tasks[task]

    def get_tasks_len(self):
        return len(self.tasks)

    def get_int_len(self):
        return len(self.intervals)

    def get_last_int(self):
        return self.intervals[-1]

    def get_before_last_int(self):
        return self.intervals[-2]

    def get_int(self, index):
        return self.intervals[index]

    def get_pair(self, task):
        return f'I,S,F:({task.task_index},{task.started},{task.finished})'

    def has_no_overlapping_intervals(self, is_letsa=True):        
        # Check for overlapping intervals
        for i in range(1, len(self.intervals)):
            if is_letsa:
                if self.intervals[i-1].started < self.intervals[i].finished:
                        # return "There are overlapping intervals, e.g. {0} vs {1}".format(self.get_pair(self.intervals[i]), self.get_pair(self.intervals[i-1]) )
                    return False
            elif self.intervals[i-1].finished > self.intervals[i].started:
                        # return "There are overlapping intervals, e.g. {0} vs {1}".format(self.get_pair(self.intervals[i]), self.get_pair(self.intervals[i-1]) )
                    return False
        # return "No overlapping intervals"
        return True

    def is_sorted(self):
        if all(self.intervals[i].started >= self.intervals[i + 1].finished for i in range(len(self.intervals) - 1)):
            return "Descending"
        elif all(self.intervals[i].finished <= self.intervals[i + 1].started for i in range(len(self.intervals) - 1)):
            return "Ascending"
        elif len(self.intervals) == 0:
            return "Some machines have no tasks assigned"
        else:
            return "Unsorted"

    def __str__(self):
        intervals_str = ', '.join([f'{self.get_pair(task)}' for task in self.intervals])
        return f'Intervals: [{intervals_str}]'
