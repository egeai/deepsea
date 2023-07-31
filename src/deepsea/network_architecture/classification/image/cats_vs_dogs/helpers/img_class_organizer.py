import os, shutil, pathlib

original_dir = pathlib.Path("/app/data/cats_vs_dogs_small/original_data")
new_base_dir = pathlib.Path("/app/data/cats_vs_dogs_small/")


def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        new_diectory = new_base_dir / subset_name / category
        os.makedirs(new_diectory)
        file_names = [f"{category}.{i}.jpg"
                  for i in range(start_index, end_index)]
        for filename in file_names:
            shutil.copyfile(src=original_dir / filename,
                            dst=new_diectory / filename)


def main():
    pass
    # make_subset("train", start_index=0, end_index=1000)
    # make_subset("validation", start_index=1000, end_index=1500)
    # make_subset("test", start_index=1500, end_index=2500)
