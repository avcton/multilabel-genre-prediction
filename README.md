# Multi-label Genre Prediction
> A genre prediction model that given a plot allows the user to predict possible genres for a movie or tv-show.

## Deployment
The app can be found at: [predict-genre.streamlit.app](https://predict-genre.streamlit.app/)

## Motivation
This model can be used to analyze synopses and identify potential mislabeling or inconsistencies in genre assignment. This can contribute to maintaining the quality and accuracy of genre metadata in media databases.

![image](https://github.com/avcton/multilabel-genre-prediction/assets/67834876/433eca81-2a62-4b72-bd80-52c3764e283f)

![CleanShot 2023-12-21 at 07 53 53@2x](https://github.com/avcton/multilabel-genre-prediction/assets/67834876/4a604c3d-8bdd-4c42-97a7-78ac58502d66)

## ML Pipeline

### Underlying Model - BiLSTM
Given a huge description, Bidirectional LSTM (BiLSTM) with the ability to process input flows in both directions can significantly help with NLP and genre context development in our case

![CleanShot 2023-12-21 at 07 57 55@2x](https://github.com/avcton/multilabel-genre-prediction/assets/67834876/832b1b28-c0a8-477f-bcd2-178d94f777c7)

### Nerual Architecture

- Input Neurons = 350 / 500
- Embedding Size = 300
- BiLSTM L1 Size = 128
- BiLSTM L2 Size = 64
- Dense Layer 1 Neurons = 64
- Dense Layer 2 Neurons = 27

## Screenshots

![CleanShot 2023-12-21 at 08 05 45@2x](https://github.com/avcton/multilabel-genre-prediction/assets/67834876/98834ac7-9ff0-4633-8e73-4405f932f168)

![CleanShot 2023-12-21 at 08 06 03@2x](https://github.com/avcton/multilabel-genre-prediction/assets/67834876/209d18f5-7512-4723-b545-efd514d537b5)

![CleanShot 2023-12-21 at 08 06 42@2x](https://github.com/avcton/multilabel-genre-prediction/assets/67834876/f105fffd-cc96-45de-90e2-5805e1697ee4)

## Collaborators / Team Members

- [Muhammad Abdullah](https://github.com/Mabdullahatif)
- Farheen Akmal

---
