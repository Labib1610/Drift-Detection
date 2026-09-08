# TASK 3b — CC-News re-probe report

- Generated: 2026-09-09T01:49:35
- Elapsed (cumulative): 1738s · scan status: **exhausted**
- Repo: `stanford-oval/ccnews` · fields: date=`published_date`, publisher=`publisher`, text=`plain_text`, lang_score≥0.9

> **Methodology caveat — read before trusting the coverage numbers.** This is a bounded scan of the *head* of each crawl-config's parquet shards (per-year row cap), and those shards are **not** shuffled by publication date. So a publisher's articles cluster into whatever date window the first shards happen to cover: the **volume** figures are informative lower bounds, but the **clean-months / max-gap** figures understate true coverage and must not be read as evidence a publisher lacks a 36-month span. Confirm coverage with a full (or shard-shuffled) extraction before ruling a language in or out on coverage grounds; rule out only on volume.

## hi

**Acceptance funnel** (one row removed per stage):

| stage | count |
| --- | --- |
| rows seen (lang-matched) | 62,011 |
| language_score ≥ 0.9 | 61,662 |
| published_date parses | 61,662 |
| publisher present | 61,662 |
| ≥50 ICU words (VALID) | **61,647** |
published_date parse rate after language filter: **100.0%**. Dropped by score<0.9: 349 (0.6% of matched).

published_date formats: `%Y-%m-%d`×61,662

language_score histogram (bucketed): 0.70:20, 0.72:16, 0.74:21, 0.76:30, 0.78:22, 0.80:29, 0.82:41, 0.84:47, 0.86:58, 0.88:65, 0.90:75, 0.92:90, 0.94:136, 0.96:306, 0.98:61,049, 1.00:6

**Crawl-year config × publication year** (valid rows):

| crawl↓ / pub→ | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2016 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 851 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2017 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 3,969 | 0 | 0 | 0 | 0 | 0 |
| 2018 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 28 | 3,974 | 0 | 0 | 0 | 0 |
| 2019 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 1 | 41 | 15,386 | 0 | 0 | 0 |
| 2020 | 1 | 2 | 2 | 2 | 4 | 1 | 7 | 3 | 5 | 20 | 221 | 14,821 | 0 | 0 |
| 2021 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 3 | 11,544 | 0 |
| 2022 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 5 | 0 | 0 | 3 | 101 | 10,637 |
Top 15 publishers by valid volume:

| publisher | articles |
| --- | --- |
| bhaskar.com | 18,485 |
| livehindustan.com | 11,262 |
| patrika.com | 3,931 |
| hindi.news18.com | 3,548 |
| navbharattimes.indiatimes.com | 3,176 |
| naidunia.jagran.com | 2,550 |
| naidunia.com | 2,303 |
| etvbharat.com | 2,288 |
| zeenews.india.com | 1,885 |
| jansatta.com | 1,680 |
| aajtak.intoday.in | 1,378 |
| hindi.oneindia.com | 1,103 |
| amarujala.com | 1,012 |
| aajtak.in | 742 |
| news.raftaar.in | 666 |
Top-3 publisher panel check (each ≥15K over ≥36 months, gap ≤2):

| publisher | articles | clean months | max gap | ok |
| --- | --- | --- | --- | --- |
| bhaskar.com | 18,485 | 3 | 58 | no |
| livehindustan.com | 11,262 | 4 | 20 | no |
| patrika.com | 3,931 | 2 | 13 | no |
**No 3-publisher panel qualifies** (on this partial scan) — candidate for substitution by Turkish/Russian from MLSUM (spec §4.3).

Secondary-topic coverage: `categories` non-empty in **76.8%**, `tags` in **95.9%** of valid rows.
Top 10 category values: `RAJYA`(5,052), `RAJASTHAN NEWS`(3,479), `MADHYA PRADESH NEWS`(3,137), `HARYANA NEWS`(1,591), `CHHATTISGARH NEWS`(1,218), `BIHAR NEWS`(968), `क्रिकेट;खेल की अन्य खबरें;इंटरव्यू;ओपीनियन;JOBS;खबरें;जनरल नॉलेज;करंट अफेयर्स;सक्सेस स्टोरी`(935), `india`(659), `other news`(655), `देश`(563)

Median ICU words/doc: **275**. Throughput: 36 matched rows/s, 35.5 valid rows/s → projected **0.8 h** to extract 100K valid docs.

## ar

**Acceptance funnel** (one row removed per stage):

| stage | count |
| --- | --- |
| rows seen (lang-matched) | 71,702 |
| language_score ≥ 0.9 | 71,587 |
| published_date parses | 71,587 |
| publisher present | 71,587 |
| ≥50 ICU words (VALID) | **71,352** |
published_date parse rate after language filter: **100.0%**. Dropped by score<0.9: 115 (0.2% of matched).

published_date formats: `%Y-%m-%d`×71,587

language_score histogram (bucketed): 0.70:8, 0.72:5, 0.74:8, 0.76:8, 0.78:9, 0.80:10, 0.82:10, 0.84:13, 0.86:20, 0.88:24, 0.90:36, 0.92:62, 0.94:185, 0.96:733, 0.98:70,571

**Crawl-year config × publication year** (valid rows):

| crawl↓ / pub→ | 1995 | 1997 | 1998 | 1999 | 2000 | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2016 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 2 | 0 | 85 | 6 | 4,789 | 1 | 0 | 0 | 0 | 0 | 0 |
| 2017 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 5 | 0 | 2 | 1 | 2 | 3 | 3 | 5,904 | 0 | 0 | 0 | 0 | 0 |
| 2018 | 0 | 0 | 214 | 0 | 46 | 9 | 13 | 8 | 0 | 0 | 1 | 0 | 0 | 1 | 2 | 0 | 0 | 1 | 9 | 0 | 0 | 127 | 8,192 | 0 | 0 | 0 | 0 |
| 2019 | 0 | 0 | 109 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 4 | 2 | 4 | 1 | 1 | 848 | 7,055 | 0 | 0 | 0 |
| 2020 | 0 | 0 | 0 | 0 | 1 | 11 | 9 | 1 | 0 | 3 | 0 | 1 | 1 | 2 | 4 | 7 | 6 | 8 | 7 | 12 | 10 | 5 | 5 | 4,279 | 13,423 | 0 | 0 |
| 2021 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 2 | 0 | 1 | 0 | 1 | 4 | 1 | 0 | 0 | 2 | 2 | 1 | 1 | 4 | 18,035 | 0 |
| 2022 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 5 | 1 | 1 | 4 | 4 | 1 | 4 | 34 | 7,973 |
Top 15 publishers by valid volume:

| publisher | articles |
| --- | --- |
| gulf365.com | 8,774 |
| gulf365.co | 8,616 |
| elbalad.news | 2,450 |
| albayan.ae | 2,309 |
| slaati.com | 2,215 |
| dotalkhaleej.co | 2,186 |
| eremnews.com | 2,047 |
| hathalyoum.net | 1,971 |
| arabic.sputniknews.com | 1,468 |
| chouftv.ma | 1,463 |
| al-marsd.com | 1,233 |
| 3yonnews.com | 1,166 |
| dostor.org | 1,151 |
| albaathmedia.sy | 1,121 |
| elwatannews.com | 1,097 |
Top-3 publisher panel check (each ≥15K over ≥36 months, gap ≤2):

| publisher | articles | clean months | max gap | ok |
| --- | --- | --- | --- | --- |
| gulf365.com | 8,774 | 1 | 0 | no |
| gulf365.co | 8,616 | 6 | 8 | no |
| elbalad.news | 2,450 | 2 | 21 | no |
**No 3-publisher panel qualifies** (on this partial scan) — candidate for substitution by Turkish/Russian from MLSUM (spec §4.3).

Secondary-topic coverage: `categories` non-empty in **72.3%**, `tags` in **75.8%** of valid rows.
Top 10 category values: `اخبار العالم`(3,557), `اخبار الرياضه`(2,395), `اخبار السعوديه`(2,360), `أخبار مصر`(2,139), `فن ومشاهير`(1,667), `الرئيسية`(1,519), `اخبار الخليج`(1,466), `آخر الأخبار`(1,061), `الاخبار المحلية`(903), `اخبار اليمن`(860)

Median ICU words/doc: **203**. Throughput: 41 matched rows/s, 41.1 valid rows/s → projected **0.7 h** to extract 100K valid docs.

## uk

**Acceptance funnel** (one row removed per stage):

| stage | count |
| --- | --- |
| rows seen (lang-matched) | 12,094 |
| language_score ≥ 0.9 | 12,065 |
| published_date parses | 12,065 |
| publisher present | 12,065 |
| ≥50 ICU words (VALID) | **11,983** |
published_date parse rate after language filter: **100.0%**. Dropped by score<0.9: 29 (0.2% of matched).

published_date formats: `%Y-%m-%d`×12,065

language_score histogram (bucketed): 0.70:1, 0.72:2, 0.76:5, 0.78:1, 0.80:2, 0.82:5, 0.84:3, 0.86:3, 0.88:7, 0.90:5, 0.92:10, 0.94:11, 0.96:13, 0.98:11,671, 1.00:355

**Crawl-year config × publication year** (valid rows):

| crawl↓ / pub→ | 2005 | 2006 | 2007 | 2008 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2016 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,913 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2017 | 0 | 1 | 0 | 0 | 0 | 0 | 67 | 0 | 0 | 0 | 0 | 1,050 | 0 | 0 | 0 | 0 | 0 |
| 2018 | 37 | 0 | 1 | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 6 | 49 | 686 | 0 | 0 | 0 | 0 |
| 2019 | 0 | 0 | 0 | 0 | 1 | 0 | 18 | 2 | 0 | 0 | 0 | 0 | 120 | 1,100 | 0 | 0 | 0 |
| 2020 | 4 | 0 | 0 | 0 | 1 | 1 | 0 | 3 | 7 | 6 | 3 | 2 | 4 | 34 | 1,213 | 0 | 0 |
| 2021 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 2 | 2,472 | 0 |
| 2022 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 2 | 3,153 |
Top 15 publishers by valid volume:

| publisher | articles |
| --- | --- |
| pravda.com.ua | 1,220 |
| 24tv.ua | 1,059 |
| rbc.ua | 870 |
| replyua.net | 737 |
| glavcom.ua | 675 |
| radiosvoboda.org | 596 |
| obozrevatel.com | 367 |
| tsn.ua | 334 |
| dzerkalo.media | 284 |
| kurs.com.ua | 212 |
| bbc.com | 195 |
| unian.ua | 194 |
| fakty.com.ua | 179 |
| focus.ua | 167 |
| zaxid.net | 161 |
Top-3 publisher panel check (each ≥15K over ≥36 months, gap ≤2):

| publisher | articles | clean months | max gap | ok |
| --- | --- | --- | --- | --- |
| pravda.com.ua | 1,220 | 2 | 44 | no |
| 24tv.ua | 1,059 | 2 | 21 | no |
| rbc.ua | 870 | 2 | 21 | no |
**No 3-publisher panel qualifies** (on this partial scan) — candidate for substitution by Turkish/Russian from MLSUM (spec §4.3).

Secondary-topic coverage: `categories` non-empty in **37.4%**, `tags` in **71.7%** of valid rows.
Top 10 category values: `Новини`(743), `Новини України`(221), `Політика`(208), `Суспільство`(117), `У світі`(115), `Україна`(104), `Новини | Міжнародні`(101), `Світ`(96), `Життя`(84), `Новини | Політика`(81)

Median ICU words/doc: **194**. Throughput: 7 matched rows/s, 6.9 valid rows/s → projected **4.0 h** to extract 100K valid docs.

## STATUS

```
GATE 0 — published_date parses at >=80% after language filtering:     PASS
GATE 5 — hi: top-3 publishers each >=15K over >=36 clean months:  FAIL
GATE 6 — ar: top-3 publishers each >=15K over >=36 clean months:  FAIL
GATE 7 — uk: top-3 publishers each >=15K over >=36 clean months:  FAIL
GATE 8 — a secondary topic field exists for >=50% of rows:            PASS

OBSERVATION — rows/second, and projected hours to extract 100K docs/language:
    hi: 36 matched/s, 35.5 valid/s -> 0.8 h
    ar: 41 matched/s, 41.1 valid/s -> 0.7 h
    uk: 7 matched/s, 6.9 valid/s -> 4.0 h
OBSERVATION — % of rows dropped by the language_score >= 0.90 filter, per language:
    hi: 0.6%
    ar: 0.2%
    uk: 0.2%

VERDICT: PROCEED WITH CAVEATS
Blockers:
  - none
Surprises worth a human decision:
  - hi: no CC-News 3-publisher panel qualifies (partial scan) — consider Turkish/Russian from MLSUM.
  - ar: no CC-News 3-publisher panel qualifies (partial scan) — consider Turkish/Russian from MLSUM.
  - uk: no CC-News 3-publisher panel qualifies (partial scan) — consider Turkish/Russian from MLSUM.
  - Scan is PARTIAL — all PASS/FAIL below are lower bounds.
```
