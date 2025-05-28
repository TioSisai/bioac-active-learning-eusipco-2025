import sys
from os import path as osp

sys.path.append(osp.dirname(osp.dirname(__file__)))
from src.al.eval import al_eval


if __name__ == "__main__":
    # Run the evaluation script
    al_eval()
