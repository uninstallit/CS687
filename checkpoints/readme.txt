# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')