
fwd_prompt = """
You are a capable agent designed to infer forward dynamics transitions in embodied decision-making. Analyze the provided images:
- The first image represents the current state of the scene.
- The subsequent four images (A, B, C, D) represent potential next states based on the given state changes.

Your task is to identify which choice (A, B, C, or D) is the most likely next state after the following state changes take place:

{STATE_CHANGES}

Focus on the visual details in the images, specifically observing the positions, placements, and relationships of the objects mentioned.

Do not perform any reasoning or explanation; simply select the letter corresponding to the most likely next state.
"""

inv_prompt = """You are a capable agent designed to infer inverse dynamics transitions in embodied decision-making. Analyze the provided images:
- The first image represents the **current state** of the scene.
- The second image represents the **next state** of the scene.

Your task is to determine which set of state changes listed below(A, B, C, or D) most accurately describes the action(s) that occurred between the two states. Each choice contains a list of state changes that describe object interactions and positional changes:


{STATE_CHANGES_CHOICES}
Focus on the visual details in the images, such as the relationships, placements, and interactions of the objects mentioned in the choices. Use this information to identify the correct sequence of state changes. Do not perform any reasoning or explanation; simply select the letter corresponding to the correct set of state changes, and only generate a single capitalized letter as your output.
"""

multi_fwd_prompt = """
You are a capable agent designed to infer multi-step forward dynamics transitions in embodied decision-making. Analyze the provided images:
- The first image represents the **current state** of the scene.
- The subsequent four options (A, B, C, D) are presented as **sequences of images** (filmstrips), each representing a potential sequence of future states.

Your task is to identify which choice (A, B, C, or D) shows the most likely sequence of states after the following actions take place **in order**:

{STATE_CHANGES}

Focus on the visual details in the images, specifically observing the positions, placements, and relationships of the objects and how they evolve through **each step** of the sequence.

Do not perform any reasoning or explanation; simply select the letter corresponding to the most likely sequence of states.
"""

multi_inv_prompt = """You are a capable agent designed to infer **multi-step inverse dynamics** transitions in embodied decision-making. Analyze the provided **sequence of images (filmstrip)**, which represents the evolution of a scene over multiple steps.

- The filmstrip shows the **entire sequence of states**, from the initial state to the final state.

Your task is to determine which set of **ordered state changes** listed below (A, B, C, or D) most accurately describes the full sequence of actions that occurred to transition through all the states shown. Each choice contains a numbered list of state changes corresponding to each step in the sequence:


{STATE_CHANGES_CHOICES}
Focus on the visual details in the images, paying close attention to how the relationships, placements, and interactions of objects change from one frame to the next. Use this information to identify the correct **sequence** of state changes. Do not perform any reasoning or explanation; simply select the letter corresponding to the correct set of state changes, and only generate a single capitalized letter as your output.
"""