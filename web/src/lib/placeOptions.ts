/** Every province, district and local unit as a place for the in-browser tools (name, centre). */
import type { PlaceOption } from "@/components/tools/PlacePicker";
import { regionByCode, regionsAt } from "./data";

export function placeOptions(): PlaceOption[] {
  return [
    ...regionsAt(1).map((r) => ({ label: `${r.name} Province`, lon: r.lon, lat: r.lat })),
    ...regionsAt(2).map((r) => ({ label: `${r.name} District`, lon: r.lon, lat: r.lat })),
    ...regionsAt(3).map((r) => ({ label: `${r.name} (${regionByCode(r.parent)?.name ?? ""} district)`, lon: r.lon, lat: r.lat })),
  ];
}
