class ListParameter():
    def __init__(self, name, range, min, max):
        self.name = name
        self.range = range
        self.min = min
        self.max = max
    
    def get_values(self):
        return [x for x in self.range if x >= self.min and x <= self.max]
        
    def increase(self):
        new_id = self.range.index(self.max)+1
        if new_id < len(self.range):
            self.max = self.range[new_id]
        
    def decrease(self):
        new_id = self.range.index(self.min)-1
        if new_id >= 0:
            self.min = self.range[new_id]

    def increase_min(self):
        new_id = self.range.index(self.min)+1
        if new_id < len(self.range):
            self.min = self.range[new_id]

    def decrease_max(self):
        new_id = self.range.index(self.max)-1
        if new_id >= 0:
            self.max = self.range[new_id]
    
    def get_config_str(self):
        str = (f'  {self.name}:\n    values:')
        for value in self.get_values():
            str+=(f'\n      - {value}')
        return str