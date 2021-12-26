from pathlib import Path
from glob import glob
import os
import shutil
from typing import List
def clean(json_structure: dict,runs_folder: Path):
    original_location = Path(os.getcwd())
    conflicts = {}
    for run in runs_folder.iterdir():
        os.chdir(str(runs_folder.joinpath(run)))
        for pattern,number in json_structure.items():
            if len(glob(pattern)) == 0:
                if run not in conflicts:
                    conflicts[run] = []
                conflicts[run].append(pattern)
        os.chdir(str(original_location))
    if len(conflicts) > 0:
        print("Conflicts:")
        for i,(run,patterns) in enumerate(conflicts.items()):
            print(f"{i}. {run} in {patterns}")
        print("Which folder to delete ? (n for none, a for all, else printer page syntax: x;y to separate x-y for intervals)")
        to_delete = input()
        folder_chosen = []
        if "a" in to_delete:
            folder_chosen = list(range(len(conflicts)))
        elif "n" in to_delete:
            folder_chosen = []
        else:
            splitted_choice = to_delete.split(";")
            for w in splitted_choice:
                if "-" in w:
                    folder_chosen.extend(list(range(int(w.split("-")[0]),int(w.split("-")[1])+1)))
                else:
                    folder_chosen.append(int(w))
        folders_to_delete: List[Path] = [list(conflicts.keys())[i] for i in folder_chosen]
        for f in folders_to_delete:
            shutil.rmtree(str(runs_folder.joinpath(f)))