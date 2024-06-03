import pandas as pd
import numpy as np 
import re
import os
import sys
import pdb
import pickle

from plots import *

# new coverage (27-5-21)

original_splits_folder = {
	'AWA2': 'AWA2/classes/',
	'CUB': 'CUB/classes/',
	'SUN': 'SUN/classes/'
}


def compute_cov(args, new_dirs, final_results, att_df):

	# coverages will be considered for all classes in dataset except unknown unknown testclasses

	_, cov_dir, _ = check_dirs(args.dataset, args, new_dirs)
	result_filename = cov_dir + 'coverage_report.txt'

	###### getting number of images of each class - change it as per folder structure of each dataset ########
	img_list = []
	img_counter = 0
	for c in att_df.index:
		class_path = final_results['al_args']['image_folder'] + '/' + c +'/'
		image_names = [f for f in os.listdir(class_path) if re.search(r'.*\.(jpg|jpeg|png)$', f)]
		img_list.append(len(image_names))

	imgs_per_class = pd.Series(data = img_list, index = att_df.index, name = 'imgs_per_class')


	with open(result_filename, 'a') as f:
		f.write('Image count per class: \n' + imgs_per_class.to_string() +'\n')
		f.write('Total images (excluding common unseen classes): ' + str(imgs_per_class.sum()) +'\n')
	###########################################################################################################

	coverage_frequency = att_df.copy()
	for c in coverage_frequency.index:
		coverage_frequency.loc[c] = (coverage_frequency.loc[c] * imgs_per_class[c])/100

	coverage_frequency = coverage_frequency.astype(int)
	total_att_presence = coverage_frequency.sum(axis = 0)
	#checking if our seen-unseen classes are indeed disjoint
	overall_test = final_results['post_swap_test'] + final_results['common_unseen']
	if len(set(final_results['post_swap_trainval']).intersection(set(overall_test))):
		print('Common classes exist! Error!')
	else:
		print('Seen-unseen disjoint')

	seen_path = original_splits_folder[args.dataset] + 'trainvalclasses.txt'
	original_trainval = open(seen_path).read().splitlines()
	orig_cov_df = coverage_frequency.loc[original_trainval]
	orig_coverage = orig_cov_df.sum(axis = 0).div(total_att_presence) * 100

	our_cov_df = coverage_frequency.loc[final_results['post_swap_trainval']] 
	our_coverage = our_cov_df.sum(axis = 0).div(total_att_presence) * 100

	results = {
		'classes': att_df.index,
		'att': att_df.columns,
		'orig_coverage': orig_coverage,
		'our_coverage': our_coverage,
	}

	pkl_name = 	cov_dir + 'coverage_results.pickle'
	pkl = open(pkl_name, 'wb')
	pickle.dump(results, pkl)
	pkl.close()

	# plot coverages
	plot_att_covs(cov_dir, pkl_name)






# splits = {
# 	'AWA2': {
# 		'our_trainval' : ['collie', 'bobcat', 'hamster', 'tiger', 'zebra', 'ox', 'horse', 'grizzly+bear', 'elephant', 'gorilla', 'weasel', 'persian+cat', 'rat', 'german+shepherd', 'rabbit', 'siamese+cat', 'otter', 'seal', 'leopard', 'giant+panda', 'bat', 'lion', 'squirrel', 'chihuahua', 'beaver', 'walrus', 'hippopotamus', 'buffalo', 'moose', 'deer', 'spider+monkey', 'skunk', 'polar+bear', 'giraffe', 'wolf', 'dalmatian', 'fox', 'pig', 'killer+whale', 'chimpanzee'],
# 		'random_trainval' : ['dalmatian', 'blue+whale', 'bobcat', 'killer+whale', 'antelope', 'german+shepherd', 'hippopotamus', 'leopard', 'spider+monkey', 'humpback+whale', 'mole', 'cow', 'squirrel', 'rat', 'lion', 'rhinoceros', 'seal', 'fox', 'tiger', 'persian+cat', 'polar+bear', 'dolphin', 'elephant', 'mouse', 'walrus', 'rabbit', 'horse', 'raccoon', 'wolf', 'sheep', 'pig', 'otter', 'gorilla', 'collie', 'zebra', 'grizzly+bear', 'buffalo', 'giant+panda', 'siamese+cat', 'hamster'],
# 		'our_test' : ['antelope', 'blue+whale', 'cow', 'dolphin', 'humpback+whale', 'mole', 'mouse', 'raccoon', 'rhinoceros', 'sheep']
# 	},

# 	'CUB': {
# 		'our_trainval' : ['159.Black_and_white_Warbler', '013.Bobolink', '171.Myrtle_Warbler', '198.Rock_Wren', '042.Vermilion_Flycatcher', '088.Western_Meadowlark', '156.White_eyed_Vireo', '047.American_Goldfinch', '134.Cape_Glossy_Starling', '149.Brown_Thrasher', '063.Ivory_Gull', '005.Crested_Auklet', '057.Rose_breasted_Grosbeak', '046.Gadwall', '019.Gray_Catbird', '025.Pelagic_Cormorant', '100.Brown_Pelican', '034.Gray_crowned_Rosy_Finch', '176.Prairie_Warbler', '189.Red_bellied_Woodpecker', '138.Tree_Swallow', '117.Clay_colored_Sparrow', '154.Red_eyed_Vireo', '133.White_throated_Sparrow', '073.Blue_Jay', '174.Palm_Warbler', '096.Hooded_Oriole', '194.Cactus_Wren', '085.Horned_Lark', '146.Forsters_Tern', '169.Magnolia_Warbler', '089.Hooded_Merganser', '075.Green_Jay', '010.Red_winged_Blackbird', '185.Bohemian_Waxwing', '066.Western_Gull', '079.Belted_Kingfisher', '048.European_Goldfinch', '083.White_breasted_Kingfisher', '129.Song_Sparrow', '035.Purple_Finch', '012.Yellow_headed_Blackbird', '188.Pileated_Woodpecker', '111.Loggerhead_Shrike', '006.Least_Auklet', '033.Yellow_billed_Cuckoo', '070.Green_Violetear', '177.Prothonotary_Warbler', '022.Chuck_will_Widow', '069.Rufous_Hummingbird', '018.Spotted_Catbird', '054.Blue_Grosbeak', '114.Black_throated_Sparrow', '061.Heermann_Gull', '016.Painted_Bunting', '036.Northern_Flicker', '115.Brewer_Sparrow', '163.Cape_May_Warbler', '049.Boat_tailed_Grackle', '003.Sooty_Albatross', '001.Black_footed_Albatross', '043.Yellow_bellied_Flycatcher', '040.Olive_sided_Flycatcher', '148.Green_tailed_Towhee', '152.Blue_headed_Vireo', '038.Great_Crested_Flycatcher', '183.Northern_Waterthrush', '157.Yellow_throated_Vireo', '126.Nelson_Sharp_tailed_Sparrow', '090.Red_breasted_Merganser', '122.Harris_Sparrow', '068.Ruby_throated_Hummingbird', '136.Barn_Swallow', '118.House_Sparrow', '039.Least_Flycatcher', '002.Laysan_Albatross', '190.Red_cockaded_Woodpecker', '175.Pine_Warbler', '184.Louisiana_Waterthrush', '197.Marsh_Wren', '098.Scott_Oriole', '128.Seaside_Sparrow', '137.Cliff_Swallow', '105.Whip_poor_Will', '028.Brown_Creeper', '056.Pine_Grosbeak', '166.Golden_winged_Warbler', '131.Vesper_Sparrow', '080.Green_Kingfisher', '180.Wilson_Warbler', '041.Scissor_tailed_Flycatcher', '067.Anna_Hummingbird', '141.Artic_Tern', '011.Rusty_Blackbird', '081.Pied_Kingfisher', '110.Geococcyx', '151.Black_capped_Vireo', '186.Cedar_Waxwing', '195.Carolina_Wren', '142.Black_Tern', '091.Mockingbird', '014.Indigo_Bunting', '032.Mangrove_Cuckoo', '017.Cardinal', '116.Chipping_Sparrow', '106.Horned_Puffin', '045.Northern_Fulmar', '179.Tennessee_Warbler', '165.Chestnut_sided_Warbler', '082.Ringed_Kingfisher', '097.Orchard_Oriole', '164.Cerulean_Warbler', '101.White_Pelican', '023.Brandt_Cormorant', '077.Tropical_Kingbird', '181.Worm_eating_Warbler', '132.White_crowned_Sparrow', '026.Bronzed_Cowbird', '192.Downy_Woodpecker', '095.Baltimore_Oriole', '193.Bewick_Wren', '015.Lazuli_Bunting', '182.Yellow_Warbler', '127.Savannah_Sparrow', '199.Winter_Wren', '135.Bank_Swallow', '162.Canada_Warbler', '112.Great_Grey_Shrike', '103.Sayornis', '076.Dark_eyed_Junco', '178.Swainson_Warbler', '172.Nashville_Warbler', '009.Brewer_Blackbird', '120.Fox_Sparrow', '094.White_breasted_Nuthatch', '074.Florida_Jay', '121.Grasshopper_Sparrow', '092.Nighthawk', '071.Long_tailed_Jaeger', '020.Yellow_breasted_Chat', '160.Black_throated_Blue_Warbler', '007.Parakeet_Auklet', '031.Black_billed_Cuckoo', '102.Western_Wood_Pewee', '027.Shiny_Cowbird', '123.Henslow_Sparrow', '170.Mourning_Warbler', '099.Ovenbird', '158.Bay_breasted_Warbler', '191.Red_headed_Woodpecker'],

# 		'random_trainval' : ['048.European_Goldfinch', '093.Clark_Nutcracker', '004.Groove_billed_Ani', '059.California_Gull', '008.Rhinoceros_Auklet', '175.Pine_Warbler', '081.Pied_Kingfisher', '134.Cape_Glossy_Starling', '116.Chipping_Sparrow', '166.Golden_winged_Warbler', '100.Brown_Pelican', '073.Blue_Jay', '027.Shiny_Cowbird', '001.Black_footed_Albatross', '096.Hooded_Oriole', '033.Yellow_billed_Cuckoo', '080.Green_Kingfisher', '111.Loggerhead_Shrike', '028.Brown_Creeper', '031.Black_billed_Cuckoo', '121.Grasshopper_Sparrow', '090.Red_breasted_Merganser', '173.Orange_crowned_Warbler', '179.Tennessee_Warbler', '143.Caspian_Tern', '142.Black_Tern', '159.Black_and_white_Warbler', '190.Red_cockaded_Woodpecker', '087.Mallard', '041.Scissor_tailed_Flycatcher', '196.House_Wren', '183.Northern_Waterthrush', '160.Black_throated_Blue_Warbler', '181.Worm_eating_Warbler', '054.Blue_Grosbeak', '062.Herring_Gull', '157.Yellow_throated_Vireo', '078.Gray_Kingbird', '002.Laysan_Albatross', '133.White_throated_Sparrow', '065.Slaty_backed_Gull', '138.Tree_Swallow', '137.Cliff_Swallow', '115.Brewer_Sparrow', '180.Wilson_Warbler', '099.Ovenbird', '162.Canada_Warbler', '009.Brewer_Blackbird', '178.Swainson_Warbler', '110.Geococcyx', '124.Le_Conte_Sparrow', '086.Pacific_Loon', '085.Horned_Lark', '186.Cedar_Waxwing', '114.Black_throated_Sparrow', '035.Purple_Finch', '057.Rose_breasted_Grosbeak', '140.Summer_Tanager', '120.Fox_Sparrow', '118.House_Sparrow', '042.Vermilion_Flycatcher', '136.Barn_Swallow', '039.Least_Flycatcher', '152.Blue_headed_Vireo', '163.Cape_May_Warbler', '007.Parakeet_Auklet', '020.Yellow_breasted_Chat', '192.Downy_Woodpecker', '172.Nashville_Warbler', '051.Horned_Grebe', '128.Seaside_Sparrow', '083.White_breasted_Kingfisher', '092.Nighthawk', '174.Palm_Warbler', '105.Whip_poor_Will', '013.Bobolink', '182.Yellow_Warbler', '058.Pigeon_Guillemot', '156.White_eyed_Vireo', '119.Field_Sparrow', '030.Fish_Crow', '149.Brown_Thrasher', '076.Dark_eyed_Junco', '193.Bewick_Wren', '082.Ringed_Kingfisher', '125.Lincoln_Sparrow', '026.Bronzed_Cowbird', '123.Henslow_Sparrow', '046.Gadwall', '098.Scott_Oriole', '056.Pine_Grosbeak', '106.Horned_Puffin', '171.Myrtle_Warbler', '053.Western_Grebe', '113.Baird_Sparrow', '170.Mourning_Warbler', '021.Eastern_Towhee', '161.Blue_winged_Warbler', '122.Harris_Sparrow', '019.Gray_Catbird', '127.Savannah_Sparrow', '070.Green_Violetear', '189.Red_bellied_Woodpecker', '197.Marsh_Wren', '147.Least_Tern', '036.Northern_Flicker', '006.Least_Auklet', '094.White_breasted_Nuthatch', '049.Boat_tailed_Grackle', '191.Red_headed_Woodpecker', '015.Lazuli_Bunting', '164.Cerulean_Warbler', '061.Heermann_Gull', '024.Red_faced_Cormorant', '130.Tree_Sparrow', '091.Mockingbird', '012.Yellow_headed_Blackbird', '079.Belted_Kingfisher', '038.Great_Crested_Flycatcher', '104.American_Pipit', '050.Eared_Grebe', '011.Rusty_Blackbird', '034.Gray_crowned_Rosy_Finch', '023.Brandt_Cormorant', '088.Western_Meadowlark', '144.Common_Tern', '155.Warbling_Vireo', '176.Prairie_Warbler', '165.Chestnut_sided_Warbler', '029.American_Crow', '132.White_crowned_Sparrow', '037.Acadian_Flycatcher', '129.Song_Sparrow', '067.Anna_Hummingbird', '063.Ivory_Gull', '018.Spotted_Catbird', '188.Pileated_Woodpecker', '117.Clay_colored_Sparrow', '045.Northern_Fulmar', '151.Black_capped_Vireo', '108.White_necked_Raven', '200.Common_Yellowthroat', '010.Red_winged_Blackbird', '052.Pied_billed_Grebe', '075.Green_Jay', '148.Green_tailed_Towhee', '074.Florida_Jay', '071.Long_tailed_Jaeger', '043.Yellow_bellied_Flycatcher', '077.Tropical_Kingbird'],

# 		'our_test' : ['004.Groove_billed_Ani', '008.Rhinoceros_Auklet', '021.Eastern_Towhee', '024.Red_faced_Cormorant', '029.American_Crow', '030.Fish_Crow', '037.Acadian_Flycatcher', '044.Frigatebird', '050.Eared_Grebe', '051.Horned_Grebe', '052.Pied_billed_Grebe', '053.Western_Grebe', '055.Evening_Grosbeak', '058.Pigeon_Guillemot', '059.California_Gull', '060.Glaucous_winged_Gull', '062.Herring_Gull', '064.Ring_billed_Gull', '065.Slaty_backed_Gull', '072.Pomarine_Jaeger', '078.Gray_Kingbird', '084.Red_legged_Kittiwake', '086.Pacific_Loon', '087.Mallard', '093.Clark_Nutcracker', '104.American_Pipit', '107.Common_Raven', '108.White_necked_Raven', '109.American_Redstart', '113.Baird_Sparrow', '119.Field_Sparrow', '124.Le_Conte_Sparrow', '125.Lincoln_Sparrow', '130.Tree_Sparrow', '139.Scarlet_Tanager', '140.Summer_Tanager', '143.Caspian_Tern', '144.Common_Tern', '145.Elegant_Tern', '147.Least_Tern', '150.Sage_Thrasher', '153.Philadelphia_Vireo', '155.Warbling_Vireo', '161.Blue_winged_Warbler', '167.Hooded_Warbler', '168.Kentucky_Warbler', '173.Orange_crowned_Warbler', '187.American_Three_toed_Woodpecker', '196.House_Wren', '200.Common_Yellowthroat']

# 	},

# 	'SUN': {
# 	'our_trainval' : ['access_road', 'airfield', 'airlock', 'airplane_cabin', 'airport_entrance', 'airport_terminal', 'airport_ticket_counter', 'alcove', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'apartment_building_outdoor', 'apse_indoor', 'apse_outdoor', 'aquarium', 'aquatic_theater', 'aqueduct', 'arch', 'archaelogical_excavation', 'archive', 'arena_basketball', 'arena_hockey', 'arena_performance', 'armory', 'arrival_gate_outdoor', 'art_gallery', 'art_school', 'art_studio', 'artists_loft', 'assembly_line', 'athletic_field_outdoor', 'atrium_home', 'atrium_public', 'attic', 'auditorium', 'auto_mechanics_indoor', 'auto_racing_paddock', 'backstage', 'badminton_court_indoor', 'badminton_court_outdoor', 'baggage_claim', 'bakery_kitchen', 'bakery_shop', 'balcony_exterior', 'balcony_interior', 'ball_pit', 'ballroom', 'bamboo_forest', 'bank_indoor', 'bank_outdoor', 'bank_vault', 'banquet_hall', 'baptistry_indoor', 'baptistry_outdoor', 'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basketball_court_indoor', 'basketball_court_outdoor', 'bathroom', 'batters_box', 'batting_cage_indoor', 'batting_cage_outdoor', 'bayou', 'bazaar_indoor', 'bazaar_outdoor', 'beach_house', 'beauty_salon', 'bedchamber', 'bedroom', 'beer_garden', 'beer_hall', 'bell_foundry', 'berth', 'betting_shop', 'bicycle_racks', 'bindery', 'bleachers_outdoor', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'booth_indoor', 'botanical_garden', 'bow_window_outdoor', 'boxing_ring', 'brewery_indoor', 'brewery_outdoor', 'brickyard_outdoor', 'bridge', 'building_complex', 'building_facade', 'bullpen', 'bullring', 'burial_chamber', 'bus_depot_outdoor', 'bus_interior', 'bus_shelter', 'bus_station_outdoor', 'butchers_shop', 'cabana', 'cabin_outdoor', 'cafeteria', 'call_center', 'campsite', 'campus', 'canal_natural', 'canteen', 'car_interior_backseat', 'car_interior_frontseat', 'cargo_deck_airplane', 'carport_freestanding', 'carport_outdoor', 'carrousel', 'casino_indoor', 'casino_outdoor', 'castle', 'catacomb', 'cathedral_outdoor', 'catwalk', 'cavern_indoor', 'cemetery', 'chalet', 'cheese_factory', 'chemical_plant', 'chemistry_lab', 'chicken_coop_indoor', 'chicken_coop_outdoor', 'chicken_farm_indoor', 'chicken_farm_outdoor', 'church_indoor', 'church_outdoor', 'circus_tent_indoor', 'circus_tent_outdoor', 'city', 'clean_room', 'cloister_indoor', 'cloister_outdoor', 'closet', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_hall', 'conference_room', 'confessional', 'construction_site', 'control_room', 'control_tower_indoor', 'control_tower_outdoor', 'corn_field', 'corral', 'cottage', 'cottage_garden', 'courtyard', 'covered_bridge_exterior', 'crawl_space', 'crevasse', 'crosswalk', 'dairy_indoor', 'dam', 'darkroom', 'day_care_center', 'departure_lounge', 'desert_road', 'desert_sand', 'diner_indoor', 'diner_outdoor', 'dinette_home', 'dinette_vehicle', 'dining_car', 'dining_hall', 'dining_room', 'dirt_track', 'discotheque', 'dolmen', 'donjon', 'doorway_indoor', 'doorway_outdoor', 'downtown', 'drainage_ditch', 'driveway', 'dry_dock', 'dugout', 'earth_fissure', 'electrical_substation', 'elevator_door', 'elevator_shaft', 'embassy', 'engine_room', 'escalator_indoor', 'escalator_outdoor', 'excavation', 'exhibition_hall', 'factory_indoor', 'factory_outdoor', 'farm', 'fence', 'ferryboat_outdoor', 'field_cultivated', 'firing_range_indoor', 'firing_range_outdoor', 'fish_farm', 'fishpond', 'fjord', 'flea_market_indoor', 'flea_market_outdoor', 'flight_of_stairs_natural', 'flight_of_stairs_urban', 'flood', 'florist_shop_indoor', 'fly_bridge', 'food_court', 'football_field', 'forest_needleleaf', 'forest_path', 'formal_garden', 'fort', 'fortress', 'foundry_indoor', 'foundry_outdoor', 'fountain', 'funeral_chapel', 'furnace_room', 'game_room', 'gangplank', 'garage_indoor', 'garage_outdoor', 'garbage_dump', 'gas_station', 'gasworks', 'gatehouse', 'general_store_outdoor', 'geodesic_dome_outdoor', 'ghost_town', 'golf_course', 'great_hall', 'greenhouse_indoor', 'greenhouse_outdoor', 'grotto', 'guardhouse', 'gulch', 'gun_deck_indoor', 'hacienda', 'hangar_indoor', 'harbor', 'hayfield', 'hedge_maze', 'hedgerow', 'heliport', 'herb_garden', 'home_office', 'home_theater', 'hoodoo', 'hospital', 'hospital_room', 'hot_spring', 'hot_tub_indoor', 'hotel_breakfast_area', 'hotel_room', 'house', 'hunting_lodge_indoor', 'hut', 'ice_floe', 'ice_shelf', 'ice_skating_rink_indoor', 'ice_skating_rink_outdoor', 'iceberg', 'igloo', 'industrial_area', 'industrial_park', 'inn_indoor', 'inn_outdoor', 'islet', 'jacuzzi_indoor', 'jail_cell', 'jail_indoor', 'japanese_garden', 'jewelry_shop', 'joss_house', 'junk_pile', 'junkyard', 'jury_box', 'kasbah', 'kennel_indoor', 'kennel_outdoor', 'kiosk_indoor', 'kiosk_outdoor', 'kitchen', 'labyrinth_indoor', 'labyrinth_outdoor', 'lake_artificial', 'landfill', 'landing_deck', 'laundromat', 'lawn', 'lean-to', 'lecture_room', 'levee', 'library_indoor', 'library_outdoor', 'lido_deck_outdoor', 'lift_bridge', 'lighthouse', 'limousine_interior', 'liquor_store_indoor', 'liquor_store_outdoor', 'loading_dock', 'lobby', 'lock_chamber', 'locker_room', 'lookout_station_outdoor', 'machine_shop', 'manhole', 'mansion', 'manufactured_home', 'market_indoor', 'market_outdoor', 'martial_arts_gym', 'mastaba', 'mausoleum', 'medina', 'military_hospital', 'mine', 'mineshaft', 'mini_golf_course_outdoor', 'moat_dry', 'moat_water', 'mobile_home', 'monastery_outdoor', 'morgue', 'mosque_indoor', 'mosque_outdoor', 'motel', 'mountain_path', 'mountain_road', 'mountain_snowy', 'movie_theater_indoor', 'movie_theater_outdoor', 'museum_indoor', 'museum_outdoor', 'music_store', 'music_studio', 'natural_history_museum', 'naval_base', 'newsstand_outdoor', 'nightclub', 'nuclear_power_plant_indoor', 'nuclear_power_plant_outdoor', 'nursery', 'nursing_home', 'oasis', 'oast_house', 'observatory_outdoor', 'office_building', 'office_cubicles', 'oil_refinery_outdoor', 'oilrig', 'operating_room', 'optician', 'orchard', 'organ_loft_exterior', 'ossuary', 'outcropping', 'overpass', 'packaging_plant', 'pagoda', 'palace', 'parade_ground', 'park', 'parking_garage_indoor', 'parking_garage_outdoor', 'parking_lot', 'parlor', 'pasture', 'patio', 'pavilion', 'pedestrian_overpass_outdoor', 'phone_booth', 'physics_laboratory', 'piano_store', 'picnic_area', 'pier', 'pig_farm', 'pilothouse_indoor', 'pilothouse_outdoor', 'pitchers_mound', 'pizzeria', 'planetarium_outdoor', 'plantation_house', 'playground', 'playroom', 'plaza', 'podium_indoor', 'podium_outdoor', 'pond', 'poolroom_establishment', 'porch', 'portico', 'power_plant_indoor', 'power_plant_outdoor', 'promenade_deck', 'pub_indoor', 'pub_outdoor', 'pulpit', 'pump_room', 'putting_green', 'quadrangle', 'quay', 'quonset_hut_outdoor', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'ramp', 'ranch', 'ranch_house', 'reading_room', 'reception', 'recreation_room', 'rectory', 'recycling_plant_indoor', 'recycling_plant_outdoor', 'repair_shop', 'residential_neighborhood', 'resort', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'restroom_indoor', 'restroom_outdoor', 'revolving_door', 'rice_paddy', 'riding_arena', 'river', 'road_cut', 'rock_arch', 'rolling_mill', 'roof', 'root_cellar', 'rope_bridge', 'roundabout', 'ruin', 'runway', 'sacristy', 'salt_plain', 'sand_trap', 'sandbar', 'sandbox', 'sauna', 'sawmill', 'schoolhouse', 'science_museum', 'seawall', 'server_room', 'shed', 'shipping_room', 'shipyard_outdoor', 'shoe_shop', 'shopfront', 'signal_box', 'skatepark', 'ski_jump', 'ski_lodge', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'spillway', 'squash_court', 'stable', 'stadium_baseball', 'stadium_football', 'stadium_outdoor', 'stadium_soccer', 'stage_indoor', 'stage_outdoor', 'staircase', 'steel_mill_indoor', 'street', 'strip_mall', 'strip_mine', 'submarine_interior', 'subway_interior', 'subway_station_corridor', 'subway_station_platform', 'sun_deck', 'supermarket', 'sushi_bar', 'swimming_hole', 'swimming_pool_indoor', 'swimming_pool_outdoor', 'synagogue_indoor', 'synagogue_outdoor', 'tea_garden', 'teashop', 'television_studio', 'temple_east_asia', 'temple_south_asia', 'temple_western', 'tennis_court_outdoor', 'tent_indoor', 'tent_outdoor', 'theater_indoor_procenium', 'theater_indoor_round', 'theater_outdoor', 'thriftshop', 'throne_room', 'ticket_booth', 'ticket_window_outdoor', 'toll_plaza', 'tollbooth', 'topiary_garden', 'tower', 'town_house', 'toyshop', 'track_outdoor', 'trading_floor', 'trailer_park', 'train_depot', 'train_railway', 'train_station_platform', 'train_station_station', 'tree_farm', 'tree_house', 'trench', 'trestle_bridge', 'tundra', 'tunnel_rail_outdoor', 'tunnel_road_outdoor', 'underwater_coral_reef', 'underwater_ice', 'underwater_kelp_forest', 'underwater_ocean_shallow', 'underwater_pool', 'underwater_wreck', 'valley', 'van_interior', 'vegetable_garden', 'velodrome_indoor', 'velodrome_outdoor', 'ventilation_shaft', 'vestry', 'veterinarians_office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano', 'volleyball_court_outdoor', 'voting_booth', 'waiting_room', 'warehouse_indoor', 'watchtower', 'water_mill', 'water_tower', 'water_treatment_plant_indoor', 'waterfall_plunge', 'weighbridge', 'wharf', 'wheat_field', 'wind_farm', 'windmill', 'window_seat', 'wine_cellar_barrel_storage', 'wine_cellar_bottle_storage', 'winery', 'witness_stand', 'workshop', 'wrestling_ring_indoor', 'yard', 'youth_hostel', 'zen_garden', 'ziggurat', 'zoo', 'anechoic_chamber', 'auto_factory', 'auto_showroom', 'bistro_indoor', 'bistro_outdoor', 'bowling_alley', 'canal_urban', 'candy_store', 'caravansary', 'cardroom', 'cliff', 'clothing_store', 'conference_center', 'cybercafe', 'dacha', 'delicatessen', 'desert_vegetation', 'dock', 'drill_rig', 'driving_range_outdoor', 'elevator_freight_elevator', 'elevator_interior', 'field_road', 'fire_escape', 'forest_broadleaf', 'forest_road', 'gazebo_exterior', 'general_store_indoor', 'geodesic_dome_indoor', 'gift_shop', 'gorge', 'hallway', 'hangar_outdoor', 'hot_tub_outdoor', 'hunting_lodge_outdoor', 'irrigation_ditch', 'jacuzzi_outdoor', 'jail_outdoor', 'lagoon', 'mesa', 'military_hut', 'observatory_indoor', 'outhouse_outdoor', 'pantry', 'pet_shop', 'planetarium_indoor', 'poolroom_home', 'promenade', 'railway_yard', 'rubble', 'schoolyard', 'shopping_mall_indoor', 'steel_mill_outdoor', 'stilt_house_water', 'tearoom', 'tennis_court_indoor', 'terrace_farm', 'theater_indoor_seats', 'track_indoor', 'train_station_outdoor', 'veranda', 'water_treatment_plant_outdoor', 'waterfall_block', 'watering_hole', 'wave'],

# 	'random_trainval' : ['skatepark', 'auditorium', 'catwalk', 'mine', 'beer_garden', 'server_room', 'donjon', 'synagogue_indoor', 'racecourse', 'mineshaft', 'railroad_track', 'schoolhouse', 'priory', 'piano_store', 'kiosk_indoor', 'pedestrian_overpass_outdoor', 'pub_indoor', 'underwater_coral_reef', 'soccer_field', 'bank_indoor', 'bathroom', 'lock_chamber', 'electrical_substation', 'weighbridge', 'van_interior', 'ruin', 'joss_house', 'lab_classroom', 'jail_outdoor', 'industrial_area', 'fjord', 'waterfall_cascade', 'hoodoo', 'nursery', 'gorge', 'wave', 'stilt_house_water', 'alcove', 'earth_fissure', 'wind_farm', 'vestry', 'ocean', 'hospital_room', 'planetarium_indoor', 'levee', 'campsite', 'stadium_soccer', 'swamp', 'estuary', 'wet_bar', 'baptistry_outdoor', 'temple_south_asia', 'theater_indoor_seats', 'monastery_outdoor', 'sauna', 'hotel_room', 'nightclub', 'brewery_outdoor', 'casino_outdoor', 'bakery_kitchen', 'park', 'plantation_house', 'raceway', 'fire_station', 'conference_center', 'pasture', 'signal_box', 'pulpit', 'chapel', 'pharmacy', 'classroom', 'kennel_outdoor', 'armory', 'art_school', 'limousine_interior', 'landfill', 'putting_green', 'schoolyard', 'veranda', 'ventilation_shaft', 'oasis', 'shower', 'darkroom', 'outcropping', 'cockpit', 'hallway', 'driveway', 'loading_dock', 'wharf', 'brewery_indoor', 'promenade', 'swimming_hole', 'inn_indoor', 'water_treatment_plant_outdoor', 'roundabout', 'industrial_park', 'sandbox', 'bedchamber', 'boxing_ring', 'shopping_mall_indoor', 'bullring', 'spillway', 'baggage_claim', 'parade_ground', 'bullpen', 'poolroom_home', 'warehouse_indoor', 'operating_room', 'circus_tent_indoor', 'wine_cellar_bottle_storage', 'tunnel_road_outdoor', 'pantry', 'manhole', 'laundromat', 'atrium_public', 'thriftshop', 'tower', 'canal_urban', 'anechoic_chamber', 'pig_farm', 'waiting_room', 'parking_garage_outdoor', 'temple_east_asia', 'flea_market_indoor', 'burial_chamber', 'garage_outdoor', 'topiary_garden', 'oilrig', 'apse_indoor', 'gazebo_exterior', 'ticket_booth', 'labyrinth_outdoor', 'forest_path', 'ski_lodge', 'fish_farm', 'butte', 'gatehouse', 'doorway_indoor', 'art_gallery', 'jail_indoor', 'control_room', 'drainage_ditch', 'water_mill', 'corridor', 'locker_room', 'lake_artificial', 'mausoleum', 'wine_cellar_barrel_storage', 'crawl_space', 'hacienda', 'roof', 'dairy_indoor', 'underwater_pool', 'residential_neighborhood', 'water_tower', 'runway', 'trench', 'velodrome_outdoor', 'grotto', 'tree_house', 'alley', 'nuclear_power_plant_indoor', 'observatory_outdoor', 'medina', 'beauty_salon', 'ball_pit', 'chicken_coop_outdoor', 'fence', 'firing_range_indoor', 'geodesic_dome_indoor', 'parlor', 'playground', 'fishpond', 'lagoon', 'youth_hostel', 'furnace_room', 'sushi_bar', 'market_outdoor', 'outhouse_outdoor', 'batting_cage_outdoor', 'beach_house', 'garbage_dump', 'butchers_shop', 'hotel_outdoor', 'firing_range_outdoor', 'forest_broadleaf', 'quay', 'formal_garden', 'throne_room', 'shed', 'resort', 'apartment_building_outdoor', 'dorm_room', 'geodesic_dome_outdoor', 'moat_water', 'dugout', 'embassy', 'balcony_exterior', 'train_station_platform', 'seawall', 'nursing_home', 'junk_pile', 'raft', 'artists_loft', 'underwater_ice', 'skyscraper', 'squash_court', 'cliff', 'flea_market_outdoor', 'bayou', 'lobby', 'hangar_indoor', 'tunnel_rail_outdoor', 'pump_room', 'windmill', 'cavern_indoor', 'military_hospital', 'overpass', 'stage_outdoor', 'delicatessen', 'banquet_hall', 'sun_deck', 'trestle_bridge', 'bookstore', 'pilothouse_outdoor', 'jewelry_shop', 'jacuzzi_indoor', 'riding_arena', 'fire_escape', 'ticket_window_outdoor', 'print_shop', 'reception', 'convenience_store_indoor', 'shoe_shop', 'watering_hole', 'workroom', 'airport_ticket_counter', 'promenade_deck', 'track_indoor', 'backstage', 'natural_history_museum', 'editing_room', 'mosque_indoor', 'morgue', 'palace', 'pilothouse_indoor', 'hut', 'kitchenette', 'martial_arts_gym', 'greenhouse_indoor', 'organ_loft_exterior', 'atrium_home', 'bus_shelter', 'sewing_room', 'lift_bridge', 'town_house', 'museum_outdoor', 'balcony_interior', 'elevator_freight_elevator', 'restaurant_kitchen', 'phone_booth', 'ranch', 'pub_outdoor', 'auto_mechanics_indoor', 'cabin_outdoor', 'elevator_interior', 'patio', 'viaduct', 'greenhouse_outdoor', 'witness_stand', 'flood', 'mountain_road', 'hunting_lodge_indoor', 'shipyard_outdoor', 'porch', 'rubble', 'pizzeria', 'kennel_indoor', 'general_store_outdoor', 'covered_bridge_exterior', 'lake_natural', 'carrousel', 'kitchen', 'auto_showroom', 'science_museum', 'newsstand_outdoor', 'teashop', 'water_treatment_plant_indoor', 'strip_mall', 'portico', 'parking_garage_indoor', 'dentists_office', 'circus_tent_outdoor', 'shopfront', 'harbor', 'repair_shop', 'trading_floor', 'great_hall', 'bus_interior', 'beer_hall', 'funeral_chapel', 'hill', 'hot_tub_outdoor', 'optician', 'chicken_coop_indoor', 'train_station_station', 'gift_shop', 'building_facade', 'ranch_house', 'ice_skating_rink_outdoor', 'crevasse', 'museum_indoor', 'botanical_garden', 'mountain_snowy', 'fly_bridge', 'courthouse', 'hot_spring', 'auto_racing_paddock', 'office', 'swimming_pool_outdoor', 'chemical_plant', 'veterinarians_office', 'workshop', 'cathedral_indoor', 'igloo', 'basketball_court_outdoor', 'bank_vault', 'garage_indoor', 'chicken_farm_indoor', 'pier', 'television_studio', 'ferryboat_outdoor', 'utility_room', 'bicycle_racks', 'zen_garden', 'corn_field', 'recycling_plant_indoor', 'quonset_hut_outdoor', 'observatory_indoor', 'bow_window_outdoor', 'planetarium_outdoor', 'elevator_lobby', 'root_cellar', 'courtyard', 'mission', 'betting_shop', 'clean_room', 'bowling_alley', 'baptistry_indoor', 'gangplank', 'arch', 'library_indoor', 'salt_plain', 'assembly_line', 'cottage', 'arena_hockey', 'dry_dock', 'castle', 'tearoom', 'athletic_field_outdoor', 'mosque_outdoor', 'departure_lounge', 'reading_room', 'cargo_deck_airplane', 'airfield', 'fortress', 'oast_house', 'call_center', 'aquatic_theater', 'podium_indoor', 'desert_vegetation', 'bell_foundry', 'fort', 'tollbooth', 'highway', 'cubicle_office', 'revolving_door', 'gun_deck_indoor', 'bar', 'cafeteria', 'watchtower', 'tree_farm', 'swimming_pool_indoor', 'exhibition_hall', 'waterfall_block', 'train_station_outdoor', 'ballroom', 'canal_natural', 'carport_outdoor', 'library_outdoor', 'house', 'cabana', 'batting_cage_indoor', 'tent_indoor', 'home_office', 'basement', 'barndoor', 'escalator_indoor', 'food_court', 'sacristy', 'slum', 'cemetery', 'abbey', 'nuclear_power_plant_outdoor', 'lean-to', 'ramp', 'heliport', 'bazaar_indoor', 'ice_floe', 'railway_yard', 'corral', 'train_depot', 'control_tower_outdoor', 'dinette_home', 'voting_booth', 'closet', 'game_room', 'field_wild', 'access_road', 'basketball_court_indoor', 'fairway', 'farm', 'pavilion', 'checkout_counter', 'factory_indoor', 'podium_outdoor', 'dinette_vehicle', 'foundry_indoor', 'dam', 'hot_tub_indoor', 'steel_mill_indoor', 'submarine_interior', 'trailer_park', 'road_cut', 'beach', 'chalet', 'airlock', 'bank_outdoor', 'mansion', 'lighthouse', 'football_field', 'galley', 'general_store_indoor', 'waterfall_cataract', 'tennis_court_indoor', 'hayfield', 'car_interior_frontseat', 'bedroom', 'dining_room', 'foundry_outdoor', 'drugstore', 'escalator_outdoor', 'gymnasium_indoor', 'church_indoor', 'street', 'amusement_park', 'booth_indoor', 'sand_trap', 'tundra', 'cheese_factory', 'pond', 'zoo', 'bus_depot_outdoor', 'particle_accelerator', 'pagoda', 'control_tower_indoor', 'home_theater', 'desert_sand', 'yard', 'temple_western', 'bistro_outdoor', 'construction_site', 'amusement_arcade', 'quadrangle', 'ice_skating_rink_indoor', 'liquor_store_indoor', 'driving_range_outdoor', 'strip_mine', 'rolling_mill', 'gulch', 'arena_basketball', 'airport_terminal', 'lookout_station_outdoor', 'cardroom', 'doorway_outdoor', 'florist_shop_indoor', 'bakery_shop', 'bamboo_forest', 'conference_room', 'batters_box', 'stone_circle', 'bow_window_indoor', 'theater_indoor_round', 'fastfood_restaurant', 'military_hut', 'diner_indoor', 'crosswalk', 'cathedral_outdoor', 'brickyard_outdoor', 'subway_interior', 'coffee_shop', 'window_seat', 'jacuzzi_outdoor', 'waterfall_plunge', 'airplane_cabin', 'basilica', 'mountain', 'iceberg', 'velodrome_indoor', 'stage_indoor', 'factory_outdoor', 'amphitheater', 'dining_car', 'theater_outdoor', 'ziggurat', 'field_road', 'archaelogical_excavation', 'aquarium', 'staircase', 'recycling_plant_outdoor', 'freeway', 'bindery', 'synagogue_outdoor', 'theater_indoor_procenium', 'parking_lot', 'bus_station_outdoor', 'train_railway', 'engine_room', 'art_studio', 'physics_laboratory', 'mini_golf_course_outdoor', 'irrigation_ditch', 'rope_bridge', 'living_room', 'tea_garden', 'jury_box', 'dirt_track', 'inn_outdoor', 'wrestling_ring_indoor', 'flight_of_stairs_urban', 'sawmill', 'lecture_room', 'restaurant_patio', 'guardhouse', 'playroom', 'dolmen', 'subway_station_platform', 'arrival_gate_outdoor', 'field_cultivated', 'ossuary', 'steel_mill_outdoor', 'candy_store', 'marsh', 'clothing_store', 'underwater_ocean_deep', 'canyon', 'stadium_outdoor', 'dock', 'airport_airport', 'caravansary', 'canteen', 'volleyball_court_outdoor', 'packaging_plant', 'downtown', 'landing_deck', 'rainforest', 'auto_factory', 'city', 'campus', 'boathouse', 'apse_outdoor', 'bridge', 'poolroom_establishment', 'aqueduct', 'picnic_area', 'tennis_court_outdoor', 'casino_indoor', 'supermarket', 'coast', 'chicken_farm_outdoor', 'badlands', 'berth', 'barn', 'videostore', 'cottage_garden', 'elevator_shaft', 'hunting_lodge_outdoor', 'cloister_indoor', 'sky', 'hotel_breakfast_area', 'liquor_store_outdoor', 'desert_road', 'jail_cell', 'office_cubicles', 'herb_garden', 'lido_deck_outdoor', 'courtroom', 'airport_entrance', 'forest_needleleaf', 'sandbar', 'attic', 'river', 'boardwalk', 'cybercafe', 'junkyard', 'underwater_wreck', 'drill_rig', 'islet', 'bazaar_outdoor', 'music_studio', 'kindergarden_classroom', 'savanna', 'day_care_center', 'underwater_kelp_forest', 'cloister_outdoor', 'restaurant', 'rice_paddy', 'shipping_room', 'tent_outdoor', 'restroom_outdoor', 'confessional', 'bistro_indoor', 'motel', 'hangar_outdoor', 'chemistry_lab', 'baseball_field', 'hospital', 'elevator_door', 'ski_slope', 'bog', 'kiosk_outdoor', 'office_building', 'toll_plaza', 'pet_shop', 'mountain_path', 'flight_of_stairs_natural', 'gas_station', 'moor', 'restroom_indoor', 'japanese_garden', 'biology_laboratory', 'childs_room', 'forest_road', 'toyshop', 'oil_refinery_outdoor'],

# 	'our_test' : ['abbey', 'airport_airport', 'badlands', 'basilica', 'beach', 'biology_laboratory', 'bog', 'bow_window_indoor', 'butte', 'canyon', 'cathedral_indoor', 'chaparral', 'chapel', 'checkout_counter', 'childs_room', 'classroom', 'convenience_store_indoor', 'corridor', 'courthouse', 'courtroom', 'creek', 'cubicle_office', 'dentists_office', 'dorm_room', 'drugstore', 'editing_room', 'elevator_lobby', 'estuary', 'fairway', 'fastfood_restaurant', 'field_wild', 'fire_station', 'freeway', 'galley', 'glacier', 'gymnasium_indoor', 'heath', 'highway', 'hill', 'hotel_outdoor', 'kindergarden_classroom', 'kitchenette', 'lab_classroom', 'lake_natural', 'living_room', 'marsh', 'mission', 'moor', 'mountain', 'newsroom', 'ocean', 'office', 'particle_accelerator', 'pharmacy', 'print_shop', 'priory', 'savanna', 'sea_cliff', 'sewing_room', 'shower', 'snowfield', 'soccer_field', 'stone_circle', 'swamp', 'underwater_ocean_deep', 'utility_room', 'waterfall_cascade', 'waterfall_cataract', 'waterfall_fan', 'wet_bar', 'woodland', 'workroom']

# 	}
# }















