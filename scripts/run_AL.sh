#!/bin/bash
cd /home/gdata/sandipan/BTP2021

# wget https://cvml.ist.ac.at/AwA2/AwA2-data.zip
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HX8ETsTbm6c0JmU-wmYafXAcvXgeimt5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HX8ETsTbm6c0JmU-wmYafXAcvXgeimt5" -O AwA2-data.zip && rm -rf /tmp/cookies.txt
# unzip AwA2-data.zip
# mkdir -p train_dir
# mkdir -p test_dir
# mkdir -p extras
# python3 complete_pipeline.py
# python3 active_learning_new.py -d 'AWA2' -t 'fox' 'bat' 'lion' 'humpback+whale' 'chimpanzee' 'elephant' 'squirrel' 'tiger' 'horse' 'seal' 'giraffe' 'rabbit' # seed for AWA2
# python3 active_learning_new.py -d 'SUN' -t 'foundry_indoor' 'railroad_track' 'biology_laboratory' 'iceberg' 'swimming_pool_outdoor' 'newsroom' 'stone_circle' 'pond' 'beach' 'excavation' 'pharmacy' 'swimming_pool_indoor' 'mountain_road' 'volcano' # seed for SUN
# python3 active_learning_new.py -d 'SUN' -t 'skatepark' 'auditorium' 'catwalk' 'mine' 'beer_garden' 'server_room' 'donjon' 'synagogue_indoor' 'racecourse' 'mineshaft' 'railroad_track' 'schoolhouse' 'priory' 'piano_store' # random seed for SUN
# python3 active_learning_new.py -d 'AWA2' -t 'dalmatian' 'blue+whale' 'bobcat' 'killer+whale' 'antelope' 'german+shepherd' 'hippopotamus' 'leopard' 'spider+monkey' 'humpback+whale' 'mole' 'cow' # random seed for AWA2
# python3 active_learning_new.py -d 'AWA2' -o 'model_new_seed' -t 'fox' 'elephant' 'ox' 'chimpanzee' 'squirrel' 'humpback+whale' 'tiger' 'giraffe' 'horse' 'seal' 'lion' 'rabbit' # new seed for AWA2
# python3 active_learning_new.py -d 'SUN' -t 'foundry_indoor' 'railroad_track' 'biology_laboratory' 'iceberg' 'newsroom' 'beach' 'funeral_chapel' 'pharmacy' 'raft' 'field_road' 'excavation' 'pond' 'mountain_road' 'swimming_pool_outdoor' 'volcano' # new seed for SUN
# python3 active_learning_new.py -d 'CUB' -o 'model_new_seed' -t '070.Green_Violetear' '067.Anna_Hummingbird' '097.Orchard_Oriole' '035.Purple_Finch' '025.Pelagic_Cormorant' '056.Pine_Grosbeak' '100.Brown_Pelican' '016.Painted_Bunting' '049.Boat_tailed_Grackle' '173.Orange_crowned_Warbler' '017.Cardinal' '069.Rufous_Hummingbird' '048.European_Goldfinch' '087.Mallard' '134.Cape_Glossy_Starling'
# python3 active_learning_new.py -d 'CUB' -o 'model_random_seed' -t '048.European_Goldfinch' '093.Clark_Nutcracker' '004.Groove_billed_Ani' '059.California_Gull' '008.Rhinoceros_Auklet' '175.Pine_Warbler' '081.Pied_Kingfisher' '134.Cape_Glossy_Starling' '116.Chipping_Sparrow' '166.Golden_winged_Warbler' '100.Brown_Pelican' '073.Blue_Jay' '027.Shiny_Cowbird' '001.Black_footed_Albatross' '096.Hooded_Oriole'
# python3 active_learning_new.py -d 'CUB' -o 'model_new_new_seed' -t '023.Brandt_Cormorant' '140.Summer_Tanager' '100.Brown_Pelican' '134.Cape_Glossy_Starling' '070.Green_Violetear' '067.Anna_Hummingbird' '083.White_breasted_Kingfisher' '035.Purple_Finch' '090.Red_breasted_Merganser' '069.Rufous_Hummingbird' '097.Orchard_Oriole' '173.Orange_crowned_Warbler' '191.Red_headed_Woodpecker' '056.Pine_Grosbeak' '016.Painted_Bunting' '049.Boat_tailed_Grackle' '136.Barn_Swallow' '157.Yellow_throated_Vireo' '025.Pelagic_Cormorant' '170.Mourning_Warbler' '017.Cardinal' '087.Mallard' '018.Spotted_Catbird' # corrected new seed for CUB
# python3 active_learning_new.py -d 'SUN' -o 'model_new_new_seed' -t 'foundry_indoor' 'railroad_track' 'biology_laboratory' 'laundromat' 'cardroom' 'iceberg' 'excavation' 'beach' 'cottage_garden' 'funeral_chapel' 'aquatic_theater' 'pharmacy' 'lagoon' 'mountain_road' 'swimming_pool_outdoor' # corrected new seed for SUN
# python3 active_learning_new.py -d 'SUN' -o 'model_new_random_seed' -t 'skatepark' 'auditorium' 'catwalk' 'mine' 'beer_garden' 'server_room' 'donjon' 'synagogue_indoor' 'racecourse' 'mineshaft' 'railroad_track' 'schoolhouse' 'priory' 'piano_store' 'kiosk_indoor' # new random seed (with 15 classes)
# python3 active_learning_new.py -d 'CUB' -o 'model_new_random_seed' -t '048.European_Goldfinch' '093.Clark_Nutcracker' '004.Groove_billed_Ani' '059.California_Gull' '008.Rhinoceros_Auklet' '175.Pine_Warbler' '081.Pied_Kingfisher' '134.Cape_Glossy_Starling' '116.Chipping_Sparrow' '166.Golden_winged_Warbler' '100.Brown_Pelican' '073.Blue_Jay' '027.Shiny_Cowbird' '001.Black_footed_Albatross' '096.Hooded_Oriole' '033.Yellow_billed_Cuckoo' '080.Green_Kingfisher' '111.Loggerhead_Shrike' '028.Brown_Creeper' '031.Black_billed_Cuckoo' '121.Grasshopper_Sparrow' '090.Red_breasted_Merganser' '173.Orange_crowned_Warbler' # new random seed(23 classes)
# https://drive.google.com/file/d/1HX8ETsTbm6c0JmU-wmYafXAcvXgeimt5/view?usp=sharing
# python3 active_learning_new.py -d 'CUB' -o 'model_new_corrected_seed' -t '100.Brown_Pelican' '016.Painted_Bunting' '070.Green_Violetear' '025.Pelagic_Cormorant' '035.Purple_Finch' '090.Red_breasted_Merganser' '097.Orchard_Oriole' '156.White_eyed_Vireo' '056.Pine_Grosbeak' '049.Boat_tailed_Grackle' '083.White_breasted_Kingfisher' '069.Rufous_Hummingbird' '017.Cardinal' '067.Anna_Hummingbird' '173.Orange_crowned_Warbler' '134.Cape_Glossy_Starling' '018.Spotted_Catbird' '123.Henslow_Sparrow' '014.Indigo_Bunting' '087.Mallard'
# python3 active_learning.py -d 'AWA2' -m 'cm_model' -t 27 -o 'cm' -r 0.7 -n 40 -s 'fox' 'bat' 'lion' 'humpback+whale' 'chimpanzee' 'elephant' 'squirrel' 'tiger' 'horse' 'seal' 'giraffe' 'rabbit'



# python3 new_active_learning_and_mat.py -d 'AWA2' -c -l 'ward' -m 'checking_kubernetes' -t 27 -o 'check' -r 0.7 -n 40 
# python3 new_active_learning_and_mat.py -d 'CUB' -c -l 'ward' -m 'checking_kubernetes' -t 100 -o 'check' -r 0.7 -n 150 
# python3 new_active_learning_and_mat.py -d 'SUN' -c -l 'ward' -m 'checking_kubernetes' -t 580 -o 'check' -r 0.7 -n 645 

# without re-clustering
# python3 new_active_learning_and_mat.py -d 'AWA2' -l 'ward' -m 'checking_kubernetes' -t 27 -o 'check' -r 0.7 -n 40 

# python3 new_active_learning_and_mat_openmax.py -d 'AWA2' -c -l 'ward' -m 'latest_trained_model' -t 27 -o 'new_seed_final' -r 0.7 -n 40 
# python3 new_active_learning_and_mat_openmax.py -d 'CUB' -c -l 'ward' -m 'latest_trained_model' -t 100 -o 'new_seed_final' -r 0.7 -n 150



# python3 new_active_learning_and_mat_openmax.py -d 'SUN' -es 25 -c -l 'ward' -lr 0.001 -m 'latest_trained_model' -t 580 -o 'new_seed_final' -r 0.7 -n 645 
# python3 new_active_learning_and_mat_openmax.py -d 'AWA2' -es 25 -c -l 'ward' -lr 0.001 -m 'latest_trained_model' -t 27 -o 'new_seed_final' -r 0.7 -n 40 
# python3 new_active_learning_and_mat_openmax.py -d 'CUB' -es 25 -c -l 'ward' -lr 0.001 -m 'latest_trained_model' -t 100 -o 'new_seed_final' -r 0.7 -n 150 



###################################################

# run commands for process involving unknown unknown testclasses

# remember to change split numbers !!!!!!!!!!!!!!!!!!!!!!!!!!

# python3 new_al_and_mat_openmax_randomTestSplits.py -d 'AWA2' -sn 1 -es 25 -c -l 'ward' -lr 0.01 -m 'latest_trained_model' -t 27 -o 'new_seed_final' -r 0.8 -n 40 -s 5 -w 0.5
# python3 new_al_and_mat_openmax_randomTestSplits.py -d 'AWA2' -sn 2 -es 25 -c -l 'ward' -lr 0.01 -m 'latest_trained_model' -t 27 -o 'new_seed_final' -r 0.8 -n 40 -s 5 -w 0.5
# python3 new_al_and_mat_openmax_randomTestSplits.py -d 'AWA2' -sn 3 -es 25 -c -l 'ward' -lr 0.01 -m 'latest_trained_model' -t 27 -o 'new_seed_final' -r 0.8 -n 40 -s 5 -w 0.5


# python3 new_al_and_mat_openmax_randomTestSplits.py -d 'CUB' -sn 1 -es 25 -c -l 'ward' -lr 0.01 -m 'latest_trained_model' -t 100 -o 'new_seed_final' -r 0.8 -n 150 -s 5 -w 0.5
# python3 new_al_and_mat_openmax_randomTestSplits_with_av_heatmap.py -d 'CUB' -sn 2 -es 25 -c -l 'ward' -lr 0.01 -m 'latest_trained_model' -t 100 -o 'new_seed_final' -r 0.8 -n 150 -s 5 -w 0.5
# python3 new_al_and_mat_openmax_randomTestSplits_with_av_heatmap.py -d 'CUB' -sn 3 -es 25 -c -l 'ward' -lr 0.01 -m 'latest_trained_model' -t 100 -o 'new_seed_final' -r 0.8 -n 150 -s 5 -w 0.5



# simultaneous SUN run
# python3 new_al_and_mat_openmax_randomTestSplits_with_av_heatmap.py -d 'SUN' -sn 2 -es 25 -c -l 'ward' -lr 0.001 -m 'latest_trained_model' -t 580 -o 'new_seed_final' -r 0.8 -n 645 -s 5 -w 0.5
# kubectl delete pod new-al


