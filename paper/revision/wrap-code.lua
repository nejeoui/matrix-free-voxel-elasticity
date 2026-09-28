-- Allow artifact paths, hashes and metric names to wrap within supplement tables.
local function wrap_code(el)
  el.text = el.text:gsub('‖', '||')
  -- ASCII equivalents retain mathematical meaning in the bundled mono font.
  el.text = el.text:gsub('ᵀ', '^T'):gsub('⁻¹', '^(-1)')
  el.text = el.text:gsub('₂', '_2'):gsub('≈', '~=')
  if FORMAT:match('latex') and #el.text > 20 then
    local words = {}
    for word in el.text:gmatch('%S+') do
      local escaped = word:gsub('[\\{}$&#_%%~%^]', function(c)
        local map = {['\\']='\\textbackslash{}', ['~']='\\textasciitilde{}', ['^']='\\textasciicircum{}'}
        return map[c] or ('\\' .. c)
      end)
      table.insert(words, '\\seqsplit{' .. escaped .. '}')
    end
    -- seqsplit discards spaces inside its argument, so split words independently.
    return pandoc.RawInline('latex', '\\texttt{' .. table.concat(words, ' ') .. '}')
  end
  return el
end

return {
  {Header = function(el)
    return el:walk({Code = function(code) return pandoc.Str(code.text) end})
  end},
  {Code = wrap_code}
}
