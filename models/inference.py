model = TabularModel(
    num_numeric_features=X_num.shape[1],
    cat_cardinalities=cat_cardinalities,
    num_classes=num_classes
)

model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
model.eval()
