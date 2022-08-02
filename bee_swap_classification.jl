using Images
using FileIO
using MLDataUtils
using Flux, Statistics, BSON, Dates, Random, Images, FileIO
using Flux: onehotbatch, onecold, crossentropy, throttle, @epochs
using Base.Iterators: partition, take
using Printf
using Plots


begin
    bee1_dir = readdir("/Users/fengkaiqi/Desktop/kaggle_bee_vs_wasp/beenew")
    #print(bee1_dir)
    wasp1_dir = readdir("/Users/fengkaiqi/Desktop/kaggle_bee_vs_wasp/waspnew")
end;
# we load the pre-proccessed images
begin
    bees1 = load.( "/Users/fengkaiqi/Desktop/kaggle_bee_vs_wasp/beenew/" .*bee1_dir)
    wasp1 = load.("/Users/fengkaiqi/Desktop/kaggle_bee_vs_wasp/waspnew/".* wasp1_dir)
end;
data = vcat(bees1, wasp1);
#Give 70% data to train and 30%data to test.
begin
    labels = vcat([0 for _ in 1:length(bees1)], [1 for _ in 1:length(wasp1)])
    (x_train, y_train), (x_test, y_test) = splitobs(shuffleobs((data, labels)), at = 0.7)
end;

# create a tuple of (bee,0) or (wasp,1)
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:1)
    return (X_batch, Y_batch)
end
#declare train set and test set
begin
    batchsize = 128
    mb_idxs = Iterators.partition(1:length(x_train), batchsize)
    train_set = [make_minibatch(x_train, y_train, i) for i in mb_idxs]
    test_set = make_minibatch(x_test, y_test, 1:length(x_test));
end;
#print(train_set)

# define our neural network model.
# you can use relu function or sigmoid function.
# Actually relu function can get a better accuracy.
model = Chain(
        Conv((3, 3), 1=>32, pad=(1,1), relu),#Conv((3, 3), 1=>32, pad=(1,1), sigmoid)
        MaxPool((2,2)),
        Conv((3, 3), 32=>64, pad=(1,1), relu),#Conv((3, 3), 32=>64, pad=(1,1), sigmoid)
        MaxPool((2,2)),
        Conv((3, 3), 64=>128, pad=(1,1), relu),#Conv((3, 3), 64=>128, pad=(1,1), sigmoid)
        MaxPool((2,2)),
        flatten,
        Dense(15488, 2),
        softmax)
begin
    train_loss = Float64[]
    test_loss = Float64[]
    acc = Float64[]
    ps = Flux.params(model)
    opt = ADAM()
    L(x, y) = Flux.crossentropy(model(x), y)
    L((x,y)) = Flux.crossentropy(model(x), y)
    accuracy(x, y, f) = mean(Flux.onecold(f(x)) .== Flux.onecold(y))
    
    function update_loss!()
        push!(train_loss, mean(L.(train_set)))
        push!(test_loss, mean(L(test_set)))
        push!(acc, accuracy(test_set..., model))  
        @printf("train loss = %.2f, test loss = %.2f, accuracy = %.2f\n", train_loss[end], test_loss[end], acc[end])
    end
end

# here we train our model for n_epochs times.
@epochs 10 Flux.train!(L, ps, train_set, opt;
               cb = Flux.throttle(update_loss!, 8))



using Plots
begin
    plot(train_loss, xlabel="Iterations", title="Model Training-sigmoid", label="Train loss", lw=2, alpha=0.9)
    plot!(test_loss, label="Test loss", lw=2, alpha=0.9)
    plot!(acc, label="Accuracy", lw=2, alpha=0.9)
end
