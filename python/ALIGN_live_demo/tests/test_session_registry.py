"""Session registry (sessions.csv) and empty-sensor checks for train_model."""
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_training
from train_model import EXPECTED_FIELDS, load_sessions


def zero_imu2_csv_bytes(sid):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(EXPECTED_FIELDS)
    for seq in range(100):
        writer.writerow([sid, 'P01', 0, seq, 1000000 + seq * 20000] + [.1] * 6 + [0.0] * 6)
    return stream.getvalue().encode('utf-8')


class SessionRegistryTests(unittest.TestCase):
    BAD_SID = '20261113_180239_b5f86fa7'

    def build(self, folder, registry_rows=None):
        data = test_training.TrainingTests.build_data(None, folder)
        (data / f'P01_{self.BAD_SID}_label0.csv').write_bytes(zero_imu2_csv_bytes(self.BAD_SID))
        if registry_rows is not None:
            lines = ['session_id,date,subject,condition,label,mount,status,note']
            lines += [f'{sid},2026-11-13,P01,body,0,unrecorded,{status},{note}'
                      for sid, status, note in registry_rows]
            (Path(folder) / 'sessions.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return data

    def test_empty_sensor_session_is_rejected_without_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.build(tmp)
            with self.assertRaisesRegex(ValueError, 'IMU2'):
                load_sessions(data, registry=None)

    def test_registry_in_parent_folder_excludes_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [(f'session{i}', 'ok', '') for i in range(4)]
            rows.append((self.BAD_SID, 'exclude', 'IMU2 all zero'))
            data = self.build(tmp, rows)
            report = {}
            sessions = load_sessions(data, report=report)
            self.assertEqual(len(sessions), 4)
            self.assertEqual(report['registry'], 'sessions.csv')
            self.assertEqual([x['session_id'] for x in report['excluded']], [self.BAD_SID])
            self.assertEqual(report['unregistered'], [])

    def test_check_and_unregistered_sessions_are_kept_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [('session0', 'check', 'loose strap'), ('session1', 'ok', ''),
                    (self.BAD_SID, 'exclude', 'IMU2 all zero')]
            data = self.build(tmp, rows)
            report = {}
            sessions = load_sessions(data, report=report)
            self.assertEqual(len(sessions), 4)
            self.assertEqual(report['check'], ['session0'])
            self.assertEqual(sorted(report['unregistered']), ['session2', 'session3'])

    def test_invalid_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.build(tmp, [('session0', 'maybe', '')])
            with self.assertRaisesRegex(ValueError, 'status'):
                load_sessions(data)


if __name__ == '__main__':
    unittest.main()
