function maybe_warn_failed_pages(xbin, warnOnTies, funcName)
if ~warnOnTies
    return
end
failed = all(isnan(xbin), 1);
nFailed = nnz(failed);
if nFailed > 0
    warning('fastinfo:SlicePageFailed', ...
        '%s could not form tie-consistent bins for %d page(s). Those page outputs were set to NaN.', ...
        funcName, nFailed);
end
end
