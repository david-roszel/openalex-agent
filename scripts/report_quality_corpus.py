"""Offline collection QC supplement; never infers human relevance or downloads text."""

import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import re
from pathlib import Path

from openalex_agent.corpus import LocalCorpus, atomic_json
from openalex_agent.dataset_integration import load_package, QUESTION_FILE
from openalex_agent.models import Paper
from openalex_agent.quality_collection import FIELD_GROUPS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--package',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh quality-analysis directory')
    a.output.mkdir(parents=True)
    manifest=json.loads((a.corpus/'manifest.json').read_text())
    if hashlib.sha256((a.corpus/'papers.jsonl').read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Corpus checksum mismatch')
    receipt=json.loads((a.corpus/'freeze-receipt.json').read_text())
    for name,checksum in receipt['files'].items():
        if hashlib.sha256((a.corpus/name).read_bytes()).hexdigest()!=checksum:raise ValueError('Frozen file checksum mismatch')
    with gzip.open(a.corpus/'raw-works.jsonl.gz','rt',encoding='utf-8') as f:raws=[json.loads(line) for line in f]
    membership=json.loads((a.corpus/'memberships.json').read_text())
    strata=defaultdict(set);disciplines=defaultdict(set)
    for wid,sid,disc,role in membership:
        strata[sid].add(wid);disciplines[wid].add(disc)
    raw_ids=[r['id'] for r in raws]
    if len(raw_ids)!=len(set(raw_ids)) or set(raw_ids)!=set(manifest['ids']):raise ValueError('Raw and normalized identities differ')
    normalized_ids=[]
    with (a.corpus/'papers.jsonl').open(encoding='utf-8') as f:
        for line in f:
            normalized=json.loads(line);normalized_ids.append(normalized['id'])
            if normalized.get('metadata_source')!='openalex':raise ValueError('Unexpected non-OpenAlex metadata')
    if len(normalized_ids)!=len(raw_ids) or set(normalized_ids)!=set(raw_ids):raise ValueError('Normalized record identities differ')
    secondary=Counter();overlap=0;assignments=[];licenses=Counter();abstract_short=0
    topic_counts=Counter();topic_names={};primary_topics=Counter();missing=Counter();year_date_disagreements=0
    doi_groups=defaultdict(list);title_groups=defaultdict(list);metadata=Counter();reference_sources=Counter()
    ids=set(raw_ids);internal_edges=set();internal_by_disc=Counter();connected=set();papers_by_disc=Counter()
    for raw in raws:
        primary=(raw.get('primary_topic') or {}).get('field',{}).get('id','').rsplit('/',1)[-1]
        fields={primary}|{(t.get('field') or {}).get('id','').rsplit('/',1)[-1] for t in raw.get('topics',[])}
        groups=sorted(d for d,ids in FIELD_GROUPS.items() if {str(i) for i in ids}&fields)
        secondary.update(groups);overlap+=len(groups)>1
        assignments.append(dict(id=raw['id'],primary_field=primary,all_topic_fields=sorted(fields),any_topic_disciplines=groups,independently_verified=False))
        license_name=(raw.get('best_oa_location') or {}).get('license')
        licenses[license_name or 'UNKNOWN']+=1
        abstract=Paper.from_openalex(raw).abstract or '';abstract_short+=len(abstract)<200
        paper=Paper.from_openalex(raw)
        for disc in disciplines[paper.id]:papers_by_disc[disc]+=1
        if paper.doi:doi_groups[paper.doi.lower()].append(dict(id=paper.id,title=paper.title,year=paper.year))
        title_key=(re.sub(r'\W+',' ',(paper.title or '').lower()).strip(),paper.year)
        title_groups[title_key].append(paper.id)
        metadata['no_authorships']+=not bool(paper.authorships)
        metadata['no_venue']+=not bool(paper.venue)
        metadata['no_outgoing_references']+=not bool(paper.references)
        metadata['missing_citation_count']+=paper.citation_count is None
        for ref in paper.references:
            if ref in ids:
                internal_edges.add((paper.id,ref));connected.update((paper.id,ref))
                for disc in disciplines[paper.id]:internal_by_disc[disc]+=1
        reference_sources['papers_with_internal_outgoing_references']+=any(ref in ids for ref in paper.references)
        for name in ('doi','publication_date','primary_topic','referenced_works','abstract_inverted_index','open_access','has_content','content_urls'):
            if name not in raw or raw[name] is None:missing[name]+=1
        primary=(raw.get('primary_topic') or {}).get('id')
        if primary:primary_topics[primary]+=1
        distinct={t.get('id') for t in raw.get('topics',[]) if t.get('id')}
        topic_counts.update(distinct)
        topic_names.update({t['id']:t.get('display_name') for t in raw.get('topics',[]) if t.get('id')})
        if raw.get('publication_date') and raw.get('publication_year'):
            year_date_disagreements+=str(raw['publication_year'])!=raw['publication_date'][:4]
    atomic_json(a.output/'taxonomy-assignments.json',assignments)
    duplicates={doi:rows for doi,rows in doi_groups.items() if len(rows)>1}
    title_duplicates=[dict(normalized_title=key[0],year=key[1],ids=rows) for key,rows in title_groups.items() if len(rows)>1]
    atomic_json(a.output/'duplicate-review.json',dict(doi_groups=duplicates,normalized_title_year_groups=title_duplicates,
        decision='PENDING human identity review; retained distinct canonical OpenAlex IDs, no automatic DOI/title merging'))
    atomic_json(a.output/'topic-coverage.json',dict(all_topic_counts=dict(topic_counts),primary_topic_counts=dict(primary_topics),topic_names=topic_names))
    package=load_package(a.package)
    dev=[r for r in package['tables'][QUESTION_FILE] if r['Preassigned split']=='DEV']
    coverage=[dict(query_id=r['Question ID'],discipline=r['Discipline'],stratum=r['Corpus stratum'],
                  acquired_stratum_papers=len(strata[r['Corpus stratum']]),question_answerability='UNVERIFIED',human_relevance_labels=0) for r in dev]
    atomic_json(a.output/'development-question-source-coverage.json',coverage)
    atomic_json(a.output/'supplement.json',dict(corpus_sha256=manifest['sha256'],raw_records=len(raws),
      any_topic_discipline_memberships=dict(secondary),papers_with_multiple_topic_disciplines=overlap,
      primary_collection_groups='See frozen quality.json; secondary counts overlap and are not extra unique papers',
      oa_location_licenses=dict(licenses),short_abstracts=abstract_short,development_questions=len(dev),
      missing_or_null_fields=dict(missing),publication_year_date_disagreements=year_date_disagreements,
      raw_normalized_id_sets_match=True,duplicate_raw_ids=0,metadata_source='openalex',
      empty_metadata_counts=dict(metadata),duplicate_doi_groups=len(duplicates),duplicate_doi_records=sum(map(len,duplicates.values())),
      duplicate_title_year_groups=len(title_duplicates),reference_source_counts=dict(reference_sources),
      internal_edges_by_source_discipline=dict(internal_by_disc),
      isolated_papers_by_discipline=dict(Counter(d for wid,groups in disciplines.items() if wid not in connected for d in groups)),
      fulltext_downloads_requested=0,fulltext_downloaded=0,ground_truth_verified=False,scientifically_validated=False,
      note='Raw and processed source metadata verified; topic assignments and source coverage are not human research judgments'))
    print(json.dumps(dict(raw_records=len(raws),overlapping_topic_disciplines=overlap,short_abstracts=abstract_short,development_questions=len(dev))))


if __name__=='__main__':main()
