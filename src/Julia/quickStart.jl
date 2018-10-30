using RDatasets, ScikitLearn
iris = dataset("datasets", "iris")

X= convert(Array, iris[[:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
y = convert(Array, iris[:Species])

@sk_import linear_model: LogisticRegression
@sk_import preprocessing: PolynomialFeatures
model = LogisticRegression(fit_intercept=true)

fit!(model, X, y)
accuracy = sum(predict(model, X) .== y) / length(y)
println("accuracy: $accuracy")
