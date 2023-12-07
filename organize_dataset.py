import os
import shutil

def organize_folders(main_folder):
    # Iterate through subfolders in the main folder
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)

        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # Create PET and GT folders in each subfolder
            pet_folder = os.path.join(subfolder_path, 'PET')
            gt_folder = os.path.join(subfolder_path, 'GT')

            if not os.path.exists(pet_folder):
                os.makedirs(pet_folder)

            if not os.path.exists(gt_folder):
                os.makedirs(gt_folder)

            # Iterate through files in the subfolder
            for file in os.listdir(subfolder_path):
                # Check for _pt.nii.gz and _ct_gtvt.nii.gz files
                if file.endswith("_pt.nii.gz"):
                    pet_source_path = os.path.join(subfolder_path, file)
                    pet_dest_path = os.path.join(pet_folder, file)
                    shutil.move(pet_source_path, pet_dest_path)
                    print(f"Moved {file} to PET folder in {subfolder}")

                elif file.endswith("_ct_gtvt.nii.gz"):
                    gt_source_path = os.path.join(subfolder_path, file)
                    gt_dest_path = os.path.join(gt_folder, file)
                    shutil.move(gt_source_path, gt_dest_path)
                    print(f"Moved {file} to GT folder in {subfolder}")

if __name__ == "__main__":
    main_folder_path = "C:/Users/Milad/Downloads/hecktor_nii_cropped"  # Replace with the actual path to your main folder
    organize_folders(main_folder_path)
