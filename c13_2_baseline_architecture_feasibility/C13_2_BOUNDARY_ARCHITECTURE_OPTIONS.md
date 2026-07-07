# C13-2 Boundary Architecture Options

Need: yes, but after strong feature pilot.

Options:

1. Focus-then-fuse localizer with modality gates over subtitle, visual, and event tokens.
2. Event-token boundary decoder with inside/outside contrast and duration-aware priors.
3. Shared-normalized multi-video span ranker trained across retrieved videos.
4. Two-stage design: strong VR candidate generator, then event-aware boundary/localizer.

Do not continue with C12-5W ranker patches. C12-5V showed that adding CONQUER hidden scalar summaries to the same ranker family regresses.

official_val_used = false
