#[derive(Clone, Debug)]
pub(crate) struct ActiveSpan {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct TaggedSentence {
    pub(crate) forms: Vec<String>,
    pub(crate) upos: Vec<String>,
    pub(crate) active: Vec<ActiveSpan>,
    pub(crate) emitted: Vec<(usize, usize)>,
}

impl TaggedSentence {
    pub(crate) fn new(tokens: Vec<(String, String)>) -> Self {
        let mut forms = Vec::with_capacity(tokens.len());
        let mut upos = Vec::with_capacity(tokens.len());
        for (form, tag) in tokens {
            forms.push(form);
            upos.push(tag);
        }
        let active = (0..forms.len())
            .map(|index| ActiveSpan {
                start: index,
                end: index + 1,
            })
            .collect();
        Self {
            forms,
            upos,
            active,
            emitted: Vec::new(),
        }
    }

    pub(crate) fn emit(&mut self, start: usize, end: usize) {
        self.emitted
            .retain(|(old_start, old_end)| !(*old_start >= start && *old_end <= end));
        self.emitted.push((start, end));
        self.emitted.sort_unstable();
    }
}
