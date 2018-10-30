using RDatasets, ScikitLearn
iris = dataset("datasets", "iris")

@sk_import naive_bayes: GaussianNB
model = GaussianNB()
X= convert(Array, iris[[:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
y = convert(Array, iris[:Species])

fit!(model, X, y)
accuracy = sum(predict(model, X) .== y) / length(y)
println("accuracy: $accuracy")
