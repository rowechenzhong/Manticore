import torch
from attention import Attention


def test_attention_fixed(decoder=False):
    encoder = Attention(size=3, internal_size=2, decoder=decoder)

    # Sanity checks
    # A = torch.tensor([[[1, 0, 0], [0, 0, 3], [0, 1, 0]]], dtype=torch.float64)
    # B = torch.tensor([[[1, 0, 0], [0, 0, 2], [0, 1, 0]]], dtype=torch.float64)

    stuff = torch.tensor(
        [[[0, 0, -1], [0, 0, 2], [0, 1, 0], [0, 1, 1]]], dtype=torch.float64)
    QW = torch.tensor(
        [[1, 2, 0], [0, -1, 1]], dtype=torch.float64)
    QB = torch.tensor(
        [0, 1], dtype=torch.float64)

    KW = torch.tensor(
        [[0, 0, 1], [2, 0, -1]], dtype=torch.float64)

    KB = torch.tensor([1, 1], dtype=torch.float64)

    VW = torch.tensor(
        [[0, 0, 1], [0, 1, 0]], dtype=torch.float64)
    VB = torch.tensor(
        [1, -2], dtype=torch.float64)

    # print(A @ B)
    encoder.query.weight = torch.nn.Parameter(QW)
    encoder.query.bias = torch.nn.Parameter(QB)

    encoder.key.weight = torch.nn.Parameter(KW)
    encoder.key.bias = torch.nn.Parameter(KB)

    encoder.value.weight = torch.nn.Parameter(VW)
    encoder.value.bias = torch.nn.Parameter(VB)

    Q = QW @ stuff.transpose(1, 2) + QB.unsqueeze(1)
    K = KW @ stuff.transpose(1, 2) + KB.unsqueeze(1)
    V = VW @ stuff.transpose(1, 2) + VB.unsqueeze(1)

    # print("Q is ", Q)
    # print("K is ", K)
    # print("V is ", V)

    # dot product
    attention_weights = Q.transpose(1, 2) @ K
    # print(attention_weights)

    # print(attention_weights)

    # normalize
    attention_weights = attention_weights / (Q.shape[2] ** 0.5)

    if decoder:
        attention_weights += torch.tensor(
            [[[0, float('-inf'), float('-inf'), float('-inf')],
              [0, 0, float('-inf'), float('-inf')],
                [0, 0, 0, float('-inf')],
                [0, 0, 0, 0]]], dtype=torch.float64
        )

    # print(attention_weights)

    # softmax
    attention_weights = torch.nn.functional.softmax(
        attention_weights, dim=2)

    expected = attention_weights @ V.transpose(1, 2)

    # print(V.transpose(1, 2))

    # print(expected)

    # print(encoder.forward(stuff))

    return encoder.forward(stuff).allclose(expected)


def test_attention(size=3, sequence_length=4, internal_size=2, batch_size=5, decoder=False):
    encoder = Attention(size=3, internal_size=2, decoder=decoder)

    # Sanity checks
    # A = torch.tensor([[[1, 0, 0], [0, 0, 3], [0, 1, 0]]], dtype=torch.float64)
    # B = torch.tensor([[[1, 0, 0], [0, 0, 2], [0, 1, 0]]], dtype=torch.float64)

    stuff = torch.rand(batch_size, sequence_length, size, dtype=torch.float64)
    QW = torch.rand(internal_size, size, dtype=torch.float64)
    QB = torch.rand(internal_size, dtype=torch.float64)

    KW = torch.rand(internal_size, size, dtype=torch.float64)
    KB = torch.rand(internal_size, dtype=torch.float64)

    VW = torch.rand(internal_size, size, dtype=torch.float64)
    VB = torch.rand(internal_size, dtype=torch.float64)

    encoder.query.weight = torch.nn.Parameter(QW)
    encoder.query.bias = torch.nn.Parameter(QB)

    encoder.key.weight = torch.nn.Parameter(KW)
    encoder.key.bias = torch.nn.Parameter(KB)

    encoder.value.weight = torch.nn.Parameter(VW)
    encoder.value.bias = torch.nn.Parameter(VB)

    Q = QW @ stuff.transpose(1, 2) + QB.unsqueeze(1)
    K = KW @ stuff.transpose(1, 2) + KB.unsqueeze(1)
    V = VW @ stuff.transpose(1, 2) + VB.unsqueeze(1)

    # print("Q is ", Q)
    # print("K is ", K)
    # print("V is ", V)

    # dot product
    attention_weights = Q.transpose(1, 2) @ K
    # print(attention_weights)

    # print(attention_weights)

    # normalize
    attention_weights = attention_weights / (Q.shape[2] ** 0.5)

    if decoder:
        indices = torch.triu_indices(
            sequence_length, sequence_length, offset=1)
        attention_weights[:, indices[0], indices[1]] = float('-inf')

    # print(attention_weights)

    # softmax
    attention_weights = torch.nn.functional.softmax(
        attention_weights, dim=2)

    expected = attention_weights @ V.transpose(1, 2)

    # print(V.transpose(1, 2))

    # print(expected)

    # print(encoder.forward(stuff))

    return encoder.forward(stuff).allclose(expected)


if __name__ == "__main__":
    assert test_attention_fixed()
    assert test_attention_fixed(decoder=True)
    assert test_attention()
    assert test_attention(decoder=True)
    print("All tests passed!")
