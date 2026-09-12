"""Merge the feed with fixtures from outside it into one weekend file."""
import pandas as pd

from predictor import fixtures

feed = fixtures.from_csv(r'D:\Downloads July 2026\SoccerData\fixtures.csv')
feed = feed[(feed.Date >= '2026-09-11') & (feed.Date <= '2026-09-13')]
raw = pd.read_csv(r'D:\Downloads July 2026\SoccerData\fixtures.csv',
                  encoding='utf-8-sig')
raw['Date'] = pd.to_datetime(raw['Date'], format='%d/%m/%Y')
raw = raw[(raw.Date >= '2026-09-11') & (raw.Date <= '2026-09-13')]

# UK kick-off times. ESPN publishes US Eastern; UK is Eastern + 5 in September.
extra = [
    # Austrian Bundesliga
    ('AUT1', '2026-09-11', '18:30', 'SV Ried', 'RB Salzburg'),
    ('AUT1', '2026-09-12', '16:00', 'SCR Altach', 'Grazer AK'),
    ('AUT1', '2026-09-12', '18:30', 'Wolfsberger AC', 'Rapid Wien'),
    ('AUT1', '2026-09-13', '13:30', 'Austria Lustenau', 'Austria Wien'),
    ('AUT1', '2026-09-13', '13:30', 'WSG Tirol', 'TSV Hartberg'),
    ('AUT1', '2026-09-13', '16:00', 'Sturm Graz', 'LASK'),
    # Eliteserien
    ('NOR1', '2026-09-12', '15:00', 'Lillestrom', 'Valerenga'),
    ('NOR1', '2026-09-12', '17:00', 'Rosenborg', 'Tromso'),
    ('NOR1', '2026-09-13', '13:30', 'Viking', 'Kristiansund'),
    ('NOR1', '2026-09-13', '16:00', 'Hamarkameratene', 'Molde'),
    ('NOR1', '2026-09-13', '16:00', 'IK Start', 'Brann'),
    ('NOR1', '2026-09-13', '16:00', 'KFUM Oslo', 'Aalesund'),
    ('NOR1', '2026-09-13', '18:15', 'Fredrikstad', 'Sarpsborg'),
    # Czech First League - kick-off times not yet published
    ('CZE1', '2026-09-12', None, 'Slovan Liberec', 'Mlada Boleslav'),
    ('CZE1', '2026-09-12', None, 'Bohemians', 'Slovacko'),
    ('CZE1', '2026-09-12', None, 'Sparta Praha', 'Jablonec'),
    ('CZE1', '2026-09-13', None, 'Banik Ostrava', 'Pardubice'),
    ('CZE1', '2026-09-13', None, 'Viktoria Plzen', 'Sigma Olomouc'),
    ('CZE1', '2026-09-13', None, 'Zlin', 'Hradec Kralove'),
    ('CZE1', '2026-09-13', None, 'Teplice', 'Slavia Praha'),
]
rows = [{'Div': d, 'Date': pd.Timestamp(dt), 'Time': t, 'HomeTeam': h,
         'AwayTeam': a} for d, dt, t, h, a in extra]

# CAF Champions League, first preliminary round, return legs.
# Leg1H / Leg1A: goals the second-leg home / away side scored in the first leg.
caf = [
    ('2026-09-12', '14:00', 'Young Africans SC (TAN)', 'Gaborone United (BOT)', 1, 1, None),
    ('2026-09-13', '14:00', 'Simba SC (TAN)', 'Mighty Wanderers (MWI)', 1, 0, None),
    ('2026-09-12', None, 'TP Mazembe (COD)', 'Medeama SC (GHA)', None, None, '11-13 Sep'),
    ('2026-09-12', None, 'Zamalek (EGY)', 'AS Port (DJI)', None, None, '11-13 Sep'),
    ('2026-09-12', None, 'Pyramids FC (EGY)', 'Gor Mahia (KEN)', None, None, '11-13 Sep'),
    ('2026-09-12', None, 'Orlando Pirates (RSA)', 'La Cure Waves (MRI)', None, None, '11-13 Sep'),
    ('2026-09-12', None, 'ASEC Mimosas (CIV)', 'ASCK (TOG)', None, None, '11-13 Sep'),
    ('2026-09-12', None, 'Al Hilal (SDN)', 'Aigle Noir (BDI)', None, None, '11-13 Sep'),
    ('2026-09-12', None, 'Atletico Petroleos (ANG)', 'UD do Songo (MOZ)', None, None, '11-13 Sep'),
]
for dt, t, h, a, l1h, l1a, note in caf:
    rows.append({'Div': 'CAFCL', 'Date': pd.Timestamp(dt), 'Time': t,
                 'HomeTeam': h, 'AwayTeam': a, 'Leg1H': l1h, 'Leg1A': l1a,
                 'WhenNote': note})

out = pd.concat([raw, pd.DataFrame(rows)], ignore_index=True)
out['Date'] = pd.to_datetime(out['Date']).dt.strftime('%d/%m/%Y')
out.to_csv('scratch/weekend-fixtures.csv', index=False)
print('feed %d + extra %d = %d fixtures' % (len(raw), len(rows), len(out)))
print(out.groupby('Div').size().sort_values(ascending=False).to_string())
