# C12-5R-B Timestamp / Boundary Label Audit

status = C12_BOUNDARY_LABEL_AUDIT_PASS

clip_length = 1.5
start_mapping = floor(start / duration * T)
end_mapping = ceil(end / duration * T) - 1
invalid_span_count = 0

Timestamp mapping is floor(start)/ceil(end)-1 with inclusive end index. Short moments often become one/few clips, but no invalid-label bug was found.

official_val_used = false
