# def get_category_name_dict():
#     """
#     Returns a list of category names for the dataset.
#     """

category_name_dict = {
    'background': 0,

    # instrument names
    'bipolar forceps': 1,
    'prograsp forceps': 2,
    'large needle driver': 3,
    'monopolar curved scissors': 4,
    'suction': 5,
    'clip applier': 6,
    'grasper': 7,
    'needle holder': 8,
    'scissors': 9,
    'forceps': 10,
    'bipolar': 11,
    'hook': 12,
    'snare': 13,
    'ligature': 14,
    'dissecting and grasping forceps': 15,  # 'stapler', 'introducer', 'catheter',

    # tissue names
    'abdominal wall': 16,
    'liver': 17,
    'gastrointestinal tract': 18,
    'fat': 19,
    'connective tissue': 20,
    'cystic duct': 21,
    'gallbladder': 22,
    'hepatic vein': 23,
    'liver ligament': 24,
    'colon': 25,
    'spleen': 26,
    'polyp': 27,
    'uterus': 28,
    # 'small intestine': 29,
    # 'stomach': 30,
}
instrument_name_dict = {
    'background': 0,
    'bipolar forceps': 1,
    'prograsp forceps': 2,
    'large needle driver': 3,
    'monopolar curved scissors': 4,
    'suction': 5,
    'clip applier': 6,
    'grasper': 7,
    'needle holder': 8,
    'scissors': 9,
    'forceps': 10,
    'bipolar': 11,
    'hook': 12,
    'snare': 13,
    'ligature': 14,
    'dissecting and grasping forceps': 15,
}
tissue_name_dict = {
    'abdominal wall': 16,
    'liver': 17,
    'gastrointestinal tract': 18,
    'fat': 19,
    'connective tissue': 20,
    'cystic duct': 21,
    'gallbladder': 22,
    'hepatic vein': 23,
    'liver ligament': 24,
    'colon': 25,
    'spleen': 26,
    'polyp': 27,
    'uterus': 28,
}
instrument_list = list(instrument_name_dict)
tissue_list = list(tissue_name_dict)

def get_category_id(category_name, use_tissue_text=True):
    category_id = category_name_dict.get(category_name, None)
    if category_id is None:
        print(f"Category name '{category_name}' not found.")

    if use_tissue_text:
        return category_id
    else:
        if category_name in instrument_name_dict:
            return instrument_name_dict[category_name]
        else:  # Mark tissue or unknown categories as ignored.
            return -1

def get_category_names(use_tissue_text=True):
    if use_tissue_text:
        return instrument_list + tissue_list
    else:
        return instrument_list
