#!/usr/bin/env bash
# ==============================================================================
# NeuronPulse Production Serverless AWS Deployment ($0.00/hr Idle · 0 EC2/GPU):
#   - Amazon CloudFront CDN (`https://d3dfklxckt5f00.cloudfront.net`, ID: E2ZSOS7XVJ55C7)
#   - Origin 1 (`*`): Private Amazon S3 Bucket (`neuronpulse-web-796330847946-us-west-2`) via OAC
#   - Origin 2 (`/api/*`): 256 MB Python 3.12 Lambda (`neuronpulse-api-origin`) in `us-west-2`
#     executing real Python ML systems passes + live Amazon Bedrock (`nova-micro` & `nova-lite`)
# ==============================================================================
set -euo pipefail

unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
export AWS_PROFILE="portfolio"

REGION="${AWS_REGION:-us-west-2}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_DIR="${PROJECT_ROOT}/web"
STAGE_DIR="/tmp/neuronpulse-aws-build"
ZIP_PATH="/tmp/neuronpulse-api.zip"
FUNC_NAME="neuronpulse-api-origin"
ROLE_NAME="QuickSpacesBedrockApiRole"
DIST_ID="${CF_DIST_ID:-E2ZSOS7XVJ55C7}"

echo "======================================================================"
echo " 1. Verifying AWS Session (Profile: ${AWS_PROFILE}, Region: ${REGION})"
echo "======================================================================"
IDENTITY_JSON="$(aws sts get-caller-identity --output json)"
ACCOUNT_ID="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["Account"])' <<< "${IDENTITY_JSON}")"
BUCKET_NAME="neuronpulse-web-${ACCOUNT_ID}-${REGION}"
echo "✅ Authenticated to AWS Account: ${ACCOUNT_ID}"

echo ""
echo "======================================================================"
echo " 2. Verifying IAM Role with Amazon Bedrock Converse Access"
echo "======================================================================"
ROLE_JSON="$(aws iam get-role --role-name "${ROLE_NAME}" --output json)"
ROLE_ARN="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["Role"]["Arn"])' <<< "${ROLE_JSON}")"

BEDROCK_POLICY='{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/*",
        "arn:aws:bedrock:*:*:inference-profile/*"
      ]
    }
  ]
}'
aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "QuickSpacesBedrockInvoke" \
  --policy-document "${BEDROCK_POLICY}"

echo ""
echo "======================================================================"
echo " 3. Packaging & Updating 256 MB Python 3.12 Lambda (${FUNC_NAME})"
echo "======================================================================"
cp "${PROJECT_ROOT}/docs/architecture.svg" "${WEB_DIR}/architecture.svg"
rm -rf "${STAGE_DIR}" "${ZIP_PATH}"
mkdir -p "${STAGE_DIR}"
cp -R "${PROJECT_ROOT}/neuronpulse" "${STAGE_DIR}/neuronpulse"
cp -R "${PROJECT_ROOT}/benchmarks" "${STAGE_DIR}/benchmarks"
cp -R "${PROJECT_ROOT}/web" "${STAGE_DIR}/web"
cp -R "${PROJECT_ROOT}/docs" "${STAGE_DIR}/docs"
find "${STAGE_DIR}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
(cd "${STAGE_DIR}" && zip -qr "${ZIP_PATH}" .)

aws lambda update-function-code \
  --function-name "${FUNC_NAME}" \
  --zip-file "fileb://${ZIP_PATH}" \
  --region "${REGION}" >/dev/null
aws lambda wait function-updated --function-name "${FUNC_NAME}" --region "${REGION}"

aws lambda update-function-configuration \
  --function-name "${FUNC_NAME}" \
  --runtime "python3.12" \
  --handler "neuronpulse.server.lambda_handler" \
  --timeout 30 \
  --memory-size 256 \
  --environment "Variables={NEURONPULSE_DELIVERY=cloudfront-s3-oac-python3.12-lambda,NEURONPULSE_BEDROCK_MODEL=us.amazon.nova-micro-v1:0}" \
  --region "${REGION}" >/dev/null
aws lambda wait function-updated --function-name "${FUNC_NAME}" --region "${REGION}"

echo "✅ Lambda ${FUNC_NAME} updated (256 MB Python 3.12)"

echo ""
echo "======================================================================"
echo " 4. Syncing Web Assets to Private S3 (${BUCKET_NAME}) & Invalidating CDN"
echo "======================================================================"
aws s3 sync "${WEB_DIR}/" "s3://${BUCKET_NAME}/" --delete --region "${REGION}"
aws cloudfront create-invalidation --distribution-id "${DIST_ID}" --paths "/*" >/dev/null

echo ""
echo "======================================================================"
echo " 🚀 NEURONPULSE v2.0 LIVE ON AMAZON CLOUDFRONT"
echo "    CloudFront URL : https://d3dfklxckt5f00.cloudfront.net"
echo "    Distribution ID: ${DIST_ID}"
echo "    Private S3 OAC : s3://${BUCKET_NAME}"
echo "    Lambda Origin  : ${FUNC_NAME} (256 MB Python 3.12)"
echo "======================================================================"
