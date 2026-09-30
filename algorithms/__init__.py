"""
algorithms - behaviour layer that decides HOW actions are performed.

    human_mouse.HumanMouse   learned mouse movement + clicks (uses controller.mouse)
    reading.read_page        scroll / pause / drift while reading a page
    human_typing.type_like_human  learned human typing (rhythm, typos, pauses, revisions)
    mouse_model/             dataset loading, training (TensorFlow) and the
                             numpy runtime sampler for the mouse model
    typing_model/            typing planner, dataset loaders, fitting / training
                             and the numpy runtime for the typing model

See ALGORITHMS.md for details.
"""
