// What the server says exists.

const ENDPOINT = "/funpack/api/modules";

export async function fetchManifest(traits) {
  const url = traits ? `${ENDPOINT}?traits=${encodeURIComponent(traits.join(","))}` : ENDPOINT;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${ENDPOINT} answered ${response.status}`);
  return response.json();
}
