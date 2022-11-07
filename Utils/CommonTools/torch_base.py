from Utils.CommonTools.timer import timer


@timer
def batch_tensor_to_device(input_, device, to_float=False):
    assert isinstance(input_, list)
    for i in range(len(input_)):
        input_[i] = input_[i].to(device)
        if to_float:
            input_[i] = input_[i].float()

    return input_
