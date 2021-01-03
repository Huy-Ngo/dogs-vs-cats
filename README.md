# Classifying dogs and cats images

In this project we build a convolutional neural network (CNN/convnet) to build a model to classify images of cats and dogs.

We use dataset [tongpython/cats-and-dog](https://www.kaggle.com/tongpython/cat-and-dog) from Kaggle.
This dataset is dedicated to public domain under [CC0 license](https://creativecommons.org/publicdomain/zero/1.0/).
This dataset contains:

- 4000 training samples for cats
- 4000 training samples for dogs
- 1000 testing samples for cats
- 1000 testing samples for dogs

We move 1000 images from training sets of each class for validation.

# Workflow

For collaborators:

Since there are several people working together, let's not push directly into main lest we step on each other's toes.

Instead, make a branch (since you're collaborators you can push directly to branches to this repo; there is no need to fork) and make PR.

You can let GitHub Actions to run the script by adding the following code to `.github/workflows/main.yml` (Or create another yml file in that folder):

```yaml
- name: The name of the task
  run: python the_script.py
```

or:

```yaml
- name: The name of the task
  run: |
    python the_script.py
    python another_script.py
```
