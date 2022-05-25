library(keras)
library(reticulate)
library(tidyverse)


original_dataset_dir <- "data/Sample/train"


base_dir <- "data/captcha_dataset" # to store a sebset of data that we are going to use
dir.create(base_dir)

train_dir <- file.path(base_dir, "train")
dir.create(train_dir)

validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)

test_dir <- file.path(base_dir, "test")
dir.create(test_dir)

train_CAPTCHA_dir <- file.path(train_dir, "CAPTCHA")
dir.create(train_CAPTCHA_dir)

validation_caPTCHA_dir <- file.path(validation_dir, "CAPTCHA")
dir.create(validation_caPTCHA_dir)

test_CAPTCHA_dir <- file.path(test_dir, "CAPTCHA")
dir.create(test_CAPTCHA_dir)

fnames <- paste0("1 (", 1:250, ")", ".png")
file.copy(file.path(original_dataset_dir, fnames), file.path(train_CAPTCHA_dir))

fnames <- paste0("1 (", 256:375, ")", ".png")
file.copy(file.path(original_dataset_dir, fnames), file.path(validation_caPTCHA_dir))

fnames <- paste0("1 (",376:500, ")", ".png")
file.copy(file.path(original_dataset_dir, fnames), file.path(test_CAPTCHA_dir))


library(keras)
model_v1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
model_v1 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("acc")
)

summary(model_v1)


train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)
train_generator <- flow_images_from_directory(
  train_dir, # Target directory
  train_datagen, # Training data generator
  target_size = c(150, 150), # Resizes all images to 150 × 150
  batch_size = 20, # 20 samples in one batch
  class_mode = "binary" # Because we use binary_crossentropy loss,
  # we need binary labels.
)
validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

history_v1 <- model_v1 %>%
  fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 30,
    validation_data =
      validation_generator,
    validation_steps = 50
  )
plot(history_v1)

model_v1 %>% save_model_hdf5("captcha_model.h5")
write_rds(history_v1, "captcha_history.rds")

datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40, # randomly rotate images up to 40 degrees
  width_shift_range = 0.2, # randomly shift 20% pictures horizontally
  height_shift_range = 0.2, # randomly shift 20% pictures vertically
  shear_range = 0.2, # randomly apply shearing transformations
  zoom_range = 0.2, # randomly zooming inside pictures
  horizontal_flip = TRUE, # randomly flipping half the images horizontally
  fill_mode = "nearest" # used for filling in newly created pixels
)
                      

fnames <- list.files(test_CAPTCHA_dir, full.names = TRUE)
img_path <- fnames[[3]] # Chooses one image to augment
img <- image_load(img_path, target_size = c(150, 150))
img_array <- image_to_array(img) # Converts the shape back to (150, 150, 3)
img_array <- array_reshape(img_array, c(1, 150, 150, 3))
augmentation_generator <- flow_images_from_data(
  img_array,
  generator = datagen,
  batch_size = 1
)
op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))
for (i in 1:4) {
  batch <- generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}
par(op)


model_v2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>% # randomly set 50% of weights to 0
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model_v2 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

test_datagen <- image_data_generator(rescale = 1/255) # no data augmentation
train_generator <- flow_images_from_directory(
  train_dir,
  datagen, # Our data augmentation configuration defined earlier
  target_size = c(150, 150),
  batch_size = 32,
  class_mode = "binary"
)
validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen, # Note that the validation data shouldn’t be augmented!
  target_size = c(150, 150),
  batch_size = 32,
  class_mode = "binary"
)
history_v2 <- model_v2 %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)

model_v2 %>% save_model_hdf5("captcha_model_v2.h5")
write_rds(history_v2, "captcha_history_v2.rds")

plot(history_v2)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen, # Note that the test data shouldn’t be augmented!
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)
model_v2 %>% evaluate(test_generator, steps = 50)


