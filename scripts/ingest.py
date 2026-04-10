import os
import gzip
import boto3
import subprocess
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── S3 Config ──────────────────────────────────────────
AWS_ACCESS_KEY     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY     = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION         = os.getenv("AWS_REGION")
S3_BUCKET          = os.getenv("AWS_S3_BUCKET")
S3_FOLDER          = "Production-Server"

# ── RDS Config ─────────────────────────────────────────
DB_HOST            = os.getenv("DB_HOST")
DB_PORT            = os.getenv("DB_PORT", "3306")
DB_NAME            = os.getenv("DB_NAME")
DB_USER            = os.getenv("DB_USER")
DB_PASSWORD        = os.getenv("DB_PASSWORD")

# ── Local temp path ────────────────────────────────────
TMP_GZ_PATH        = "/tmp/latest_backup.sql.gz"
TMP_SQL_PATH       = "/tmp/latest_backup.sql"


def get_latest_backup_key():
    """Find the latest backup file in S3 based on last modified time."""
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )

    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_FOLDER + "/")
    files = response.get("Contents", [])

    if not files:
        raise Exception("No backup files found in S3 bucket!")

    # Sort by last modified and pick the latest
    latest = sorted(files, key=lambda x: x["LastModified"], reverse=True)[0]
    print(f"✅ Latest backup found: {latest['Key']} (modified: {latest['LastModified']})")
    return latest["Key"]


def download_from_s3(key):
    """Download the backup file from S3."""
    print(f"⬇️  Downloading {key} from S3...")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
    s3.download_file(S3_BUCKET, key, TMP_GZ_PATH)
    print(f"✅ Downloaded to {TMP_GZ_PATH}")


def decompress_backup():
    """Decompress the .sql.gz file to .sql."""
    print("🗜️  Decompressing backup...")
    with gzip.open(TMP_GZ_PATH, "rb") as gz_file:
        with open(TMP_SQL_PATH, "wb") as sql_file:
            sql_file.write(gz_file.read())
    print(f"✅ Decompressed to {TMP_SQL_PATH}")


def import_to_rds():
    """Import the SQL file into RDS MySQL."""
    print(f"📥 Importing SQL into RDS database '{DB_NAME}'...")
    command = (
        f"mysql -h {DB_HOST} -P {DB_PORT} -u {DB_USER} "
        f"-p{DB_PASSWORD} {DB_NAME} < {TMP_SQL_PATH}"
    )
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"MySQL import failed:\n{result.stderr}")

    print("✅ Data successfully imported into RDS!")


def cleanup():
    """Remove temp files."""
    for path in [TMP_GZ_PATH, TMP_SQL_PATH]:
        if os.path.exists(path):
            os.remove(path)
    print("🧹 Temp files cleaned up")


def main():
    print("=" * 50)
    print(f"🚀 Starting ingestion at {datetime.now()}")
    print("=" * 50)
    try:
        key = get_latest_backup_key()
        download_from_s3(key)
        decompress_backup()
        import_to_rds()
        cleanup()
        print("=" * 50)
        print(f"🎉 Ingestion completed at {datetime.now()}")
        print("=" * 50)
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        cleanup()
        raise


if __name__ == "__main__":
    main()