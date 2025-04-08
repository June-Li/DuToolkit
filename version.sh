#!/bin/sh

# Get the version from the environment variable
VERSION=${OCR_VERSION:?"VERSION environment variable is not set"}
# Check if VERSION conforms to Poetry versioning standards
if ! echo "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+)?$'; then
    # If it doesn't conform, prepend with a fixed prefix
    VERSION="0.1.0"
    echo "Version does not conform to Poetry standards. Adjusted to: $VERSION"
fi


# Update version.py
echo "version = \"$VERSION\"" > version.py

# Update pyproject.toml
sed -i "s/^version = .*/version = \"$VERSION\"/" pyproject.toml

echo "Version updated to $VERSION in version.py and pyproject.toml"
