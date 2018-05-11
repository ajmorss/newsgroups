def create_model_mn_factory():
    with C.layers.default_options(initial_state=0.1):
        m = C.layers.Sequential([
            C.layers.Recurrence(C.layers.LSTM(50), go_backwards=False),
            C.layers.Recurrence(C.layers.LSTM(50), go_backwards=False),
            C.layers.Dense(lookahead, name='output')
        ])
        return C.sequence.last(m)