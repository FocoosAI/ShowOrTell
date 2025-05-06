#!/bin/bash

# ANSI color codes for prettier output
GREEN='\e[32m'
BLUE='\e[34m'
YELLOW='\e[33m'
CYAN='\e[36m'
MAGENTA='\e[35m'
RED='\e[31m'
BOLD='\e[1m'
UNDERLINE='\e[4m'
RESET='\e[0m'

# Function to print section headers
print_header() {
    echo -e "\n${BOLD}${BLUE}====== $1 ======${RESET}\n"
}

# Function to print task messages
print_task() {
    echo -e "${YELLOW}➜ ${CYAN}$1${RESET}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${RESET}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗ $1${RESET}"
}

download_houseparts() {
    print_header "House-Parts Dataset"
    print_task "Creating directory and downloading dataset..."
    mkdir -p House-Parts
    
    curl -L "https://universe.roboflow.com/ds/QP6BRYqwhd?key=eKnDwNtwGF" > House-Parts/houseparts.zip
    print_task "Unzipping House-Parts dataset..."
    unzip -q House-Parts/houseparts.zip -d House-Parts
    rm House-Parts/houseparts.zip
    
    print_task "Converting annotations to mask format..."
    python3 coco_to_mask.py --base-path ./House-Parts \
          --splits train valid \
          --annotation-paths train/_annotations.coco.json valid/_annotations.coco.json
    print_success "House-Parts dataset processing completed!"
}

download_pizza() {
    print_header "Pizza Dataset"
    print_task "Creating directory and downloading dataset..."
    mkdir -p Pizza
    
    curl -L "https://universe.roboflow.com/ds/OFYM25lwrJ?key=Xf0YFgLpDE" > Pizza/pizza.zip
    print_task "Unzipping Pizza dataset..."
    unzip -q Pizza/pizza.zip -d Pizza
    rm Pizza/pizza.zip
    
    print_task "Converting annotations to mask format..."
    python3 coco_to_mask.py --base-path ./Pizza \
          --splits train valid \
          --annotation-paths train/_annotations.coco.json valid/_annotations.coco.json
    print_success "Pizza dataset processing completed!"
}

download_toolkits() {
    print_header "Toolkits Dataset"
    print_task "Creating directory and downloading dataset..."
    mkdir -p Toolkits
    
    curl -L "https://universe.roboflow.com/ds/aRlQYI2Aqj?key=3jVdHAMvXr" > Toolkits/toolkits.zip
    print_task "Unzipping Toolkits dataset..."
    unzip -q Toolkits/toolkits.zip -d Toolkits
    rm Toolkits/toolkits.zip
    
    print_task "Converting annotations to mask format..."
    python3 coco_to_mask.py --base-path ./Toolkits \
          --splits train valid \
          --annotation-paths train/_annotations.coco.json valid/_annotations.coco.json
    print_success "Toolkits dataset processing completed!"
}

download_trash() {
    print_header "Trash Dataset"
    print_task "Creating directory and downloading dataset..."
    mkdir -p Trash
    
    curl -L "https://universe.roboflow.com/ds/pYfDQeThKL?key=DH1LvoKyEq" > Trash/trash.zip
    print_task "Unzipping Trash dataset..."
    unzip -q Trash/trash.zip -d Trash
    rm Trash/trash.zip
    
    print_task "Converting annotations to mask format..."
    python3 coco_to_mask.py --base-path ./Trash \
          --splits train valid \
          --annotation-paths train/_annotations.coco.json valid/_annotations.coco.json
    print_success "Trash dataset processing completed!"
}

download_loveda() {
    print_header "LoveDA Dataset"
    print_task "Creating directory and downloading dataset..."
    mkdir -p LoveDA
    
    wget -q --show-progress -O "LoveDA/Train.zip" "https://zenodo.org/records/5706578/files/Train.zip"
    wget -q --show-progress -O "LoveDA/Val.zip" "https://zenodo.org/records/5706578/files/Val.zip"
    print_task "Unzipping LoveDA dataset..."
    unzip -q LoveDA/Train.zip -d LoveDA && rm LoveDA/Train.zip
    unzip -q LoveDA/Val.zip -d LoveDA && rm LoveDA/Val.zip
    
    print_task "Preprocessing dataset..."
    python3 preprocess_loveda.py --base_path ./LoveDA
    print_success "LoveDA dataset processing completed!"
}

download_zerowaste() {
    print_header "ZeroWaste Dataset"
    print_task "Downloading dataset..."
    
    wget -q --show-progress -O "zerowaste-f-final.zip" "https://zenodo.org/records/6412647/files/zerowaste-f-final.zip"
    print_task "Unzipping ZeroWaste dataset..."
    unzip -q zerowaste-f-final.zip && mv splits_final_deblurred ZeroWaste/ && rm zerowaste-f-final.zip
    
    print_task "Preprocessing dataset..."
    python3 preprocess_zerowaste.py --base_path ./ZeroWaste
    print_success "ZeroWaste dataset processing completed!"
}

download_mhpv1() {
    print_header "MHPv1 Dataset"
    print_task "Downloading dataset..."
    
    python3 -m gdown 1hTS8QJBuGdcppFAr_bvW2tsD9hW_ptr5 -O MHPv1.zip
    print_task "Unzipping MHPv1 dataset..."
    unzip -q MHPv1.zip && mv LV-MHP-v1 MHPv1/ && rm MHPv1.zip
    
    print_task "Preprocessing dataset..."
    python3 preprocess_mhp.py --base_path ./MHPv1
    print_success "MHPv1 dataset processing completed!"
}

download_pidray() {
    print_header "PIDray Dataset"
    print_task "Creating directory and downloading dataset..."
    mkdir -p PIDray
    
    python3 -m gdown --folder https://drive.google.com/drive/folders/1zvMIc1bqteRN9Z36hHYpoTGoZArsh4mE -O PIDray
    if [ $? -ne 0 ]; then
        print_error "Error: Failed to download PIDray dataset. Please manually download the dataset from https://drive.google.com/drive/folders/1zvMIc1bqteRN9Z36hHYpoTGoZArsh4mE"
        exit 1
    fi
    rm PIDray/model_weights.zip
    
    print_task "Unzipping PIDray dataset..."
    for file in PIDray/*.tar.gz; do
        if [ -f "$file" ]; then
            echo "Unzipping $file..."
            tar -xzf "$file" -C PIDray/ && rm "$file"
        fi
    done
    
    print_task "Preprocessing dataset..."
    python3 preprocess_pidray.py --base_path ./PIDray
    print_success "PIDray dataset processing completed!"
}

download_uecfood() {
    print_header "UECFOODPIX Dataset"
    print_task "Downloading dataset..."
    
    wget -q --show-progress -O "UECFOODPIXCOMPLETE.tar" "https://mm.cs.uec.ac.jp/uecfoodpix/UECFOODPIXCOMPLETE.tar"
    print_task "Unzipping UECFOODPIX dataset..."
    tar -xf UECFOODPIXCOMPLETE.tar && rm UECFOODPIXCOMPLETE.tar
    mv UECFOODPIXCOMPLETE UECFOOD

    print_task "Organizing directory structure..."
    mv UECFOOD/data/UECFoodPIXCOMPLETE/* UECFOOD/
    mv UECFOOD/data/category.txt UECFOOD/
    rm -rf UECFOOD/data
    
    print_task "Preprocessing dataset..."
    python3 preprocess_uecfood.py --base_path ./UECFOOD
    print_success "UECFOOD dataset processing completed!"
}

download_pascalvoc() {
    print_header "PascalVOC Dataset"
    print_task "Downloading dataset..."
    
    wget -q --show-progress -O "VOCdevkit_11-May-2012.tar" "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    print_task "Unzipping PascalVOC dataset..."
    tar -xf VOCdevkit_11-May-2012.tar && rm VOCdevkit_11-May-2012.tar
    print_success "PascalVOC dataset processing completed!"
}

download_ade20k() {
    print_header "ADE20K Dataset"
    print_task "Downloading dataset..."
    
    wget -q --show-progress "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    print_task "Unzipping ADE20K dataset..."
    unzip -q ADEChallengeData2016.zip && rm ADEChallengeData2016.zip
    print_success "ADE20K dataset processing completed!"
}

download_cityscapes() {
    print_header "Cityscapes Dataset"
    print_task "Creating directory and preparing for download..."
    mkdir -p cityscapes

    echo -e "${YELLOW}⚠ ${BOLD}Cityscapes dataset requires login credentials.${RESET}"
    echo -e "${CYAN}Enter your credentials below (will be saved in cookies and removed after download).${RESET}"
    read -p "$(echo -e "${MAGENTA}Enter your Cityscapes email: ${RESET}")" EMAIL
    read -s -p "$(echo -e "${MAGENTA}Enter your Cityscapes password: ${RESET}")" PASSWORD
    echo
    
    EMAIL_ESCAPED=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$EMAIL'))")
    PASSWORD_ESCAPED=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$PASSWORD'))")
    print_task "Logging in to Cityscapes website..."
    wget -q --keep-session-cookies --save-cookies=cityscapes/cookies.txt --post-data "username=$EMAIL_ESCAPED&password=$PASSWORD_ESCAPED&submit=Login" -O /dev/null https://www.cityscapes-dataset.com/login/

    print_task "Downloading images and annotations..."
    wget --load-cookies cityscapes/cookies.txt -q --show-progress -O "cityscapes/gtFine_trainvaltest.zip" --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
    
    wget --load-cookies cityscapes/cookies.txt -q --show-progress -O "cityscapes/leftImg8bit_trainvaltest.zip" --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
    rm cityscapes/cookies.txt

    print_task "Unzipping Cityscapes dataset files..."
    unzip -q cityscapes/gtFine_trainvaltest.zip -d cityscapes && rm cityscapes/gtFine_trainvaltest.zip  
    unzip -o -q cityscapes/leftImg8bit_trainvaltest.zip -d cityscapes && rm cityscapes/leftImg8bit_trainvaltest.zip

    print_task "Creating train ID label images..."
    CITYSCAPES_DATASET=./cityscapes python3 -m cityscapesscripts.preparation.createTrainIdLabelImgs
    print_success "\nCityscapes dataset processing completed!"
}

download_uavid() {
    print_header "UAVid Dataset"
    echo -e "${YELLOW}⚠ ${BOLD}Note: UAVid dataset requires registration at${RESET} ${UNDERLINE}https://uavid.nl/#download${RESET}"
    echo -e "${CYAN}Please visit the website and follow their download instructions${RESET}"
    echo -e "${CYAN}Once downloaded, please place the dataset in the ./uavid directory${RESET}"
    echo -e "${CYAN}Then run the following command to preprocess the dataset:${RESET}"
    echo -e "${MAGENTA}python3 preprocess_uavid.py --base_path ./uavid${RESET}"
}

# Main function to download all datasets
download_all() {
    print_header "Starting Download of All Datasets"
    download_pascalvoc
    download_ade20k
    download_houseparts
    download_pizza
    download_toolkits
    download_trash
    download_loveda
    download_zerowaste
    download_mhpv1
    download_uecfood
    download_uavid
    download_cityscapes
    download_pidray

    echo -e "\n${BOLD}${GREEN}========================================${RESET}"
    echo -e "${BOLD}${GREEN}    All downloads completed successfully!${RESET}"
    echo -e "${BOLD}${GREEN}========================================${RESET}\n"
}

show_help() {
    echo -e "${BOLD}${BLUE}Usage:${RESET} $0 [OPTION]"
    echo -e "${BOLD}${CYAN}Download benchmark datasets for few-shot semantic segmentation.${RESET}"
    echo
    echo -e "${BOLD}${BLUE}Options:${RESET}"
    echo -e "  ${YELLOW}--all${RESET}             Download all datasets"
    echo -e "  ${YELLOW}--cityscapes${RESET}      Download Cityscapes dataset"
    echo -e "  ${YELLOW}--ade20k${RESET}          Download ADE20K dataset"
    echo -e "  ${YELLOW}--pascalvoc${RESET}       Download PascalVOC dataset"
    echo -e "  ${YELLOW}--houseparts${RESET}      Download House-Parts dataset"
    echo -e "  ${YELLOW}--pizza${RESET}           Download Pizza dataset"
    echo -e "  ${YELLOW}--toolkits${RESET}        Download Toolkits dataset"
    echo -e "  ${YELLOW}--trash${RESET}           Download Trash dataset"
    echo -e "  ${YELLOW}--loveda${RESET}          Download LoveDA dataset"
    echo -e "  ${YELLOW}--zerowaste${RESET}       Download ZeroWaste dataset"
    echo -e "  ${YELLOW}--mhpv1${RESET}           Download MHPv1 dataset"
    echo -e "  ${YELLOW}--pidray${RESET}          Download PIDray dataset"
    echo -e "  ${YELLOW}--uecfood${RESET}         Download UECFOODPIX dataset"
    echo -e "  ${YELLOW}--uavid${RESET}           Show UAVid dataset information"
    echo -e "  ${YELLOW}--help${RESET}            Display this help message"
    echo
    echo -e "${CYAN}If no options are provided, all datasets will be downloaded.${RESET}"
}

if [ $# -eq 0 ]; then
    download_all
else
    for arg in "$@"; do
        case $arg in
            --all)
                download_all
                exit 0
                ;;
            --pascalvoc)
                download_pascalvoc
                ;;
            --ade20k)
                download_ade20k
                ;;
            --cityscapes)
                download_cityscapes
                ;;
            --houseparts)
                download_houseparts
                ;;
            --pizza)
                download_pizza
                ;;
            --toolkits)
                download_toolkits
                ;;
            --trash)
                download_trash
                ;;
            --loveda)
                download_loveda
                ;;
            --zerowaste)
                download_zerowaste
                ;;
            --mhpv1)
                download_mhpv1
                ;;
            --pidray)
                download_pidray
                ;;
            --uecfood)
                download_uecfood
                ;;
            --uavid)
                download_uavid
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $arg${RESET}"
                show_help
                exit 1
        esac
    done
    echo -e "\n${BOLD}${GREEN}Download process completed!${RESET}\n"
fi