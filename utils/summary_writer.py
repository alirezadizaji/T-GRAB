import os

from tensorboardX import SummaryWriter


def init_summary_writer(filewriter_path):
    if os.path.exists(filewriter_path):
        import shutil
        shutil.rmtree(filewriter_path)
        
    os.makedirs(filewriter_path, exist_ok=True)

    writer = SummaryWriter(filewriter_path, comment='visualizer')
    return writer
