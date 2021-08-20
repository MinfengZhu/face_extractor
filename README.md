# Face Extractor
Face Extractor for images and videos

# Install
conda env create --prefix ./env --file environment.yml
conda activate ./env
pip install imageio-ffmpeg
pip install face-alignment
pip install ipyfilechooser
git clone https://github.com/adipandas/multi-object-tracker
cd multi-object-tracker
pip install -r requirements.txt
pip install -e .
cd ..
sudo rm -r multi-object-tracker
