module CNN

function ReLU(z)
    max(0, z)
end

function softmax(z)
    exponents = exp(z)
    exponents / sum(exponents)
end

function crossentropy(pred, label)
    -1 * sum(label * log(pred))
end

end