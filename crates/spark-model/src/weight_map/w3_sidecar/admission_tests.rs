// SPDX-License-Identifier: AGPL-3.0-only

use super::admission::W3SidecarRequest;
#[cfg(unix)]
use std::os::unix::ffi::OsStringExt;

const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn request(
    layers: Option<&str>,
    path: Option<&str>,
    sha: Option<&str>,
    size: Option<&str>,
) -> anyhow::Result<Option<W3SidecarRequest>> {
    W3SidecarRequest::from_values(64, layers, path, sha, size)
}

#[test]
fn all_empty_is_disabled_but_every_partial_request_fails() {
    assert!(request(None, None, None, None).unwrap().is_none());
    assert!(
        request(Some(""), Some(""), Some(""), Some(""))
            .unwrap()
            .is_none()
    );
    let valid = [
        Some("1"),
        Some("/model/w3.safetensors"),
        Some(SHA),
        Some("42"),
    ];
    for missing in 0..valid.len() {
        let mut fields = valid;
        fields[missing] = None;
        assert!(request(fields[0], fields[1], fields[2], fields[3]).is_err());
    }
}

#[test]
fn strict_layer_set_is_sorted_unique_and_bounded() {
    let admitted = request(
        Some("5,7-10,12"),
        Some("/model/w3.safetensors"),
        Some(SHA),
        Some("3743457280"),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        admitted.layers().iter().copied().collect::<Vec<_>>(),
        [5, 7, 8, 9, 10, 12]
    );
    assert_eq!(admitted.path().to_str(), Some("/model/w3.safetensors"));
    assert_eq!(admitted.sha256()[0..4], [0x01, 0x23, 0x45, 0x67]);
    assert_eq!(admitted.size(), 3_743_457_280);

    for invalid in [
        "", "1,", ",1", "1,,2", "01", "+1", " 1", "1 ", "1-", "-1", "1-2-3", "3-2", "1,1", "1-3,2",
        "64", "63-64",
    ] {
        assert!(
            request(
                Some(invalid),
                Some("/model/w3.safetensors"),
                Some(SHA),
                Some("42")
            )
            .is_err(),
            "{invalid:?}"
        );
    }
}

#[test]
fn artifact_identity_fields_are_canonical() {
    for bad_path in ["w3.safetensors", "./w3.safetensors", ""] {
        assert!(request(Some("1"), Some(bad_path), Some(SHA), Some("42")).is_err());
    }
    for bad_sha in [
        "",
        "0",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcde",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0",
        "0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF",
        "g123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ] {
        assert!(
            request(
                Some("1"),
                Some("/model/w3.safetensors"),
                Some(bad_sha),
                Some("42")
            )
            .is_err(),
            "{bad_sha:?}"
        );
    }
    for bad_size in [
        "",
        "0",
        "00",
        "01",
        "+1",
        " 1",
        "1 ",
        "18446744073709551616",
    ] {
        assert!(
            request(
                Some("1"),
                Some("/model/w3.safetensors"),
                Some(SHA),
                Some(bad_size)
            )
            .is_err(),
            "{bad_size:?}"
        );
    }
}

#[test]
fn disabled_is_geometry_independent_but_enabled_geometry_is_exactly_bounded() {
    assert!(
        W3SidecarRequest::from_values(0, None, None, None, None)
            .unwrap()
            .is_none()
    );
    for total_layers in [0, 65, usize::MAX] {
        assert!(
            W3SidecarRequest::from_values(
                total_layers,
                Some("1"),
                Some("/model/w3.safetensors"),
                Some(SHA),
                Some("42")
            )
            .is_err()
        );
    }
}

#[cfg(unix)]
#[test]
fn non_utf8_environment_fields_are_rejected_not_treated_as_absent() {
    let malformed = std::ffi::OsString::from_vec(vec![0xff]);
    for field in 0..4 {
        let valid = [
            std::ffi::OsStr::new("1"),
            std::ffi::OsStr::new("/model/w3.safetensors"),
            std::ffi::OsStr::new(SHA),
            std::ffi::OsStr::new("42"),
        ];
        let mut values = valid.map(Some);
        values[field] = Some(malformed.as_os_str());
        assert!(
            W3SidecarRequest::from_os_values(64, values[0], values[1], values[2], values[3])
                .is_err()
        );
    }
}
