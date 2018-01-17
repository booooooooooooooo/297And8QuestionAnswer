FROM tensorflow/tensorflow
RUN pip install nltk
RUN python -m nltk.downloader --dir=/usr/local/share/nltk_data perluniprops punkt
RUN pip install tqdm
RUN pip install -U floyd-cli



WORKDIR /297And8QuestionAnswer
