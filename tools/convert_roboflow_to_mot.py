import json
import os
import argparse

def convert_roboflow_to_mot(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Filter categories: Keep only 'person' and map it to ID 1
    person_cat_id = None
    for cat in data['categories']:
        if cat['name'] == 'person':
            person_cat_id = cat['id']
            break
    
    if person_cat_id is None:
        print(f"Error: 'person' category not found in {json_path}")
        return

    new_categories = [{"id": 1, "name": "person", "supercategory": "none"}]
    
    # 2. Update Images: Ensure video_id and frame_id exist
    # Roboflow might not provide video_id/frame_id for detection datasets.
    # We will assume all images belong to video_id=1 and frame_id is sequential or derived from filename.
    
    new_images = []
    image_id_map = {} # old_id -> new_id (if needed, but we keep original IDs usually)
    
    for i, img in enumerate(data['images']):
        new_img = img.copy()
        
        # Add video_id if missing
        if 'video_id' not in new_img:
            new_img['video_id'] = 1
            
        # Add frame_id if missing. Try to parse from filename or use index
        if 'frame_id' not in new_img:
            # Example filename: frame_000231_jpg.rf.xyz.jpg -> 231
            try:
                # Simple heuristic: look for numbers in filename
                import re
                # matches numbers at the start or after 'frame_'
                match = re.search(r'frame_(\d+)', new_img['file_name'])
                if match:
                    new_img['frame_id'] = int(match.group(1))
                else:
                    new_img['frame_id'] = i + 1
            except:
                new_img['frame_id'] = i + 1
        
        new_images.append(new_img)

    # 3. Update Annotations: Filter non-person and update category_id
    new_annotations = []
    for ann in data['annotations']:
        if ann['category_id'] == person_cat_id:
            new_ann = ann.copy()
            new_ann['category_id'] = 1 # Force to 1
            
            # Ensure track_id exists (for detection, it might be -1 or missing)
            if 'track_id' not in new_ann:
                new_ann['track_id'] = -1
                
            new_annotations.append(new_ann)

    # Construct new JSON
    new_data = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": new_categories,
        "images": new_images,
        "annotations": new_annotations
    }

    with open(output_path, 'w') as f:
        json.dump(new_data, f)
    
    print(f"Converted {json_path} to {output_path}")
    print(f"  - Original categories: {data['categories']}")
    print(f"  - New categories: {new_categories}")
    print(f"  - Images: {len(new_images)}")
    print(f"  - Annotations: {len(new_annotations)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Roboflow COCO JSON to MOT format for OC-SORT")
    parser.add_argument("input_json", help="Path to input Roboflow JSON")
    parser.add_argument("output_json", help="Path to output JSON")
    args = parser.parse_args()

    convert_roboflow_to_mot(args.input_json, args.output_json)
