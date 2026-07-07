# C15-0 Protocol

status = `C15_PROTOCOL_READY`

- branch: `c15-boundary-localizer-redesign`
- commit: `c619bbad76c45acc49f89c0e1a1f6778a4258fa9`
- current promoted system: `C7-B6 R1SelectiveTop1`
- C14 decision accepted: `True`
- raw video exists: `false`
- C14 text status: `C14_TEXT_SUBTITLE_HARMFUL`
- C14 event status: `C14_EVENT_FEATURE_HARMFUL`
- C12 localizer/feature builder reusable: `True`
- missing core artifacts: `[]`

Correction to the instruction: raw TVR video is not locally available, so C15-2
uses existing clip-level visual energy, subtitle features, query embeddings, and
C12 localizer logits. It does not fake raw-video features.

official was not run.
