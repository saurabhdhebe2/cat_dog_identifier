from PIL import Image
import os

def verify_images(directory):
    """Verify and remove invalid image files from a directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            img = Image.open(file_path)
            img.verify()  # Verify the integrity of the image
        except (IOError, SyntaxError) as e:
            print(f"Removing bad file: {file_path}")
            os.remove(file_path)  # Remove the bad file

if __name__ == '__main__':
    # Paths to the dataset
    cat_dir = 'data/train/cats'
    dog_dir = 'data/train/dogs'
    
    print("Verifying cat images...")
    verify_images(cat_dir)
    print("Verifying dog images...")
    verify_images(dog_dir)

    val_cat_dir = 'data/val/cats'
    val_dog_dir = 'data/val/dogs'
    
    print("Verifying  val cat images...")
    verify_images(val_cat_dir)
    print("Verifying val dog images...")
    verify_images(val_dog_dir)

    print("Image verification complete.")
