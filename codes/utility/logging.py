import os
from datetime import datetime
import pathlib

class Logger():
    def __init__(self, path,is_debug,target="log",path2=None,ablation_target=None):
        pathlib.Path(f"{path}").mkdir(parents=True,exist_ok=True)
        self.target=target
        self.path = path
        self.log_ = is_debug
        self.path2=path2
        self.ablation_target=ablation_target

        self.logging("#"*30+"   New Logger Start   "+"#"*30)
    def logging(self, s):
        s = str(s)
        print(datetime.now().strftime('%Y-%m-%d-%H:%M:'), s)
        if self.log_:
            with open(os.path.join(self.path,f"{self.target}.txt"), 'a+') as f_log:
                f_log.write(str(datetime.now().strftime('%Y-%m-%d %H:%M:')) + s + '\n')
            if self.path2:
                with open(os.path.join(self.path2, f"{self.target}.txt"), 'a+') as f_log:
                    f_log.write(str(datetime.now().strftime('%Y-%m-%d %H:%M:')) + s + '\n')

    def logging_sum(self,s):
        if self.path2:
            print(s)
            with open(os.path.join(self.path2, f"sum_{str(self.ablation_target)}.txt"), 'a+') as f_log:
                f_log.write(s + '\n')